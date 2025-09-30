import argparse
import logging
from openai import OpenAI
from tqdm import tqdm
import string
import os
import time
import logging
from tqdm import tqdm
import ir_datasets
import json
from datasets import DownloadManager
from collections import defaultdict
import gzip
import pickle
from pathlib import Path
import requests
import sys
from datasets import load_dataset
import os
import statistics
from collections import Counter


IRDS_PREFIX = "irds:"
HFG_PREFIX = "hfds:"

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File + console handlers
file_handler = logging.FileHandler("convert_query.log")
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

def read_queries(queries_path: str, split: str = None, lower_case: bool = False, text_fields=["text"]):
    queries = []
    if queries_path.startswith(IRDS_PREFIX):
        irds_name = queries_path.replace(IRDS_PREFIX, "")
        dataset = ir_datasets.load(irds_name)
        print(dataset)
        for query in tqdm(
            dataset.queries_iter(),
            desc=f"Loading queries from ir_datasets: {queries_path}",
        ):
            query_id = query.query_id
            texts = [getattr(query, field) for field in text_fields]
            text = " ".join(texts)
            text = text.lower() if lower_case else text
            queries.append((query_id, text))
    elif queries_path.startswith(HFG_PREFIX):
        hfg_name = queries_path.replace(HFG_PREFIX, "")
        dataset = load_dataset(hfg_name, split,trust_remote_code=True) if split else load_dataset(hfg_name,trust_remote_code=True)
        split = "passage" if split == None else split
        for row in tqdm(
            dataset[split],
            desc=f"Loading data from HuggingFace datasets: {hfg_name}",
        ):
            text = row["text"]
            text = text.lower() if lower_case else text
            queries.append((row["id"] if "id" in row else row["_id"], text))
    else:
        with open(queries_path, "r") as f:
            for line in tqdm(f, desc=f"Reading queries from {queries_path}"):
                query_id, text = line.strip().split("\t")
                text = text.lower() if lower_case else text
                queries.append((query_id, text))
    return queries

def normalize(text):
    """Remove punctuation, whitespace, lowercase (for validation)."""
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = "".join(text.split()).lower()
    return text


def convert_to_sentence_case(query, model, base_url, api_key, max_retries=3):
    """Convert a query to sentence case using the OpenAI API with retries."""
    client = OpenAI(base_url=base_url, api_key=api_key)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a text processor. Your task is to convert text to sentence case, "
                            "while only capitalizing proper nouns (names, places, organizations, etc.) "
                            "and the first word of the sentence. "
                            "Do not change spelling, grammar, or word order. "
                            "Do not add or remove words. "
                            "Do not add or remove punctuation. "
                            "Do not add any punctuation under any circumstances. "
                            "Do not translate the text. Keep it in English. "
                            "Do not remove or alter white space between words. "
                            "Only adjust capitalization where necessary.\n\n"
                            "Example:\n"
                            "Input: who is the aziz hashim\n"
                            "Output: Who is the Aziz Hashim"
                        ),
                    },
                    {"role": "user", "content": query},
                ],
            )
            converted = response.choices[0].message.content.strip()

            if normalize(query) == normalize(converted):
                return converted

        except Exception as e:
            logger.error(f"Error during API call: {e}, retrying...")
            time.sleep(5)  # shorter wait to avoid blocking forever

    logger.error(f"Failed after retries, keeping original query: {query}")
    return query


def load_existing_results(output_path):
    """Load already processed queries into a dict {id: converted_query}."""
    if not output_path or not os.path.exists(output_path):
        return {}
    results = {}
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                qid, query = parts
                results[qid] = query
    return results


def append_results(output_path, buffer):
    """Append buffered results to file."""
    with open(output_path, "a", encoding="utf-8") as f_out:
        for pid, conv in buffer:
            f_out.write(f"{pid}\t{conv}\n")


def process_dataset(dataset_name, split, model, base_url, api_key, output_path=None, save_every=100):
    """
    Process all queries in the dataset:
    - Skip queries already in output_path
    - Convert them to sentence case via LLM (one by one)
    - Save progress periodically
    """
    dataset = read_queries(dataset_name, split=split)
    existing_results = load_existing_results(output_path)

    buffer = []
    total, skipped, processed = 0, 0, 0

    iterator = tqdm(dataset, desc="Processing Queries")

    for qid, query in iterator:
        total += 1
        if qid in existing_results:
            skipped += 1
            continue

        converted = convert_to_sentence_case(query, model, base_url, api_key)
        buffer.append((qid, converted))
        processed += 1

        # Save every N queries
        if output_path and processed % save_every == 0:
            append_results(output_path, buffer)
            buffer = []
            logger.info(f"Progress saved ({processed} processed so far)")

    # Final flush
    if output_path and buffer:
        append_results(output_path, buffer)

    logger.info(f"Total queries: {total}, skipped: {skipped}, newly processed: {processed}")
    return None if output_path else buffer

def generate_latex_case_table(output_path):
    """Generate a concise LaTeX table comparing uncased vs cased queries."""
    from collections import Counter
    import statistics
    import os

    uncased_queries = []
    cased_queries = []

    uncased_word_counts = []
    cased_word_counts = []

    uncased_words = []
    cased_words = []

    if not os.path.exists(output_path):
        logger.warning(f"No output file found at {output_path}. Cannot generate LaTeX table.")
        return

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) != 2:
                continue
            _, query = parts

            words = query.split()
            if query.islower():
                uncased_queries.append(query)
                uncased_word_counts.append(len(words))
                uncased_words.extend([w.lower() for w in words])
            else:
                cased_queries.append(query)
                cased_word_counts.append(len(words))
                cased_words.extend([w.lower() for w in words])

    total_queries = len(uncased_queries) + len(cased_queries)

    def safe_mean(lst):
        return statistics.mean(lst) if lst else 0

    latex_table = f"""
\\begin{{table}}[h!]
\\centering
\\begin{{tabular}}{{l c c c}}
\\hline
Statistic & Uncased & Cased & Combined \\\\
\\hline
Total queries & {len(uncased_queries)} & {len(cased_queries)} & {total_queries} \\\\
Percentage & {len(uncased_queries)/total_queries*100:.1f}\\% & {len(cased_queries)/total_queries*100:.1f}\\% & 100\\% \\\\
Average words & {safe_mean(uncased_word_counts):.2f} & {safe_mean(cased_word_counts):.2f} & {safe_mean(uncased_word_counts + cased_word_counts):.2f} \\\\
\\hline
\\end{{tabular}}
\\caption{{Concise case statistics for converted queries}}
\\label{{tab:query_case_stats}}
\\end{{table}}
"""
    print(latex_table)
    return latex_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert IR dataset queries to sentence case via LLM API.")
    parser.add_argument("--dataset", type=str, default="irds:msmarco-passage/train/judged", help="Dataset name or file path")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--output", type=str, default="data/converted_queries.tsv", help="Path to save results")
    parser.add_argument("--model", type=str, default="llama-3-8b-instruct", help="LLM model name")
    parser.add_argument("--key", type=str, default=None, help="API key (default: use IDA_LLM_API_KEY env var)")
    parser.add_argument("--base_url", type=str, default="http://api.llm.apps.os.dcs.gla.ac.uk/v1", help="LLM API base URL")
    parser.add_argument("--save_every", type=int, default=100, help="Save progress every N queries")

    args = parser.parse_args()

    process_dataset(
        dataset_name=args.dataset,
        split=args.split,
        model=args.model,
        base_url=args.base_url,
        api_key=args.key if args.key else os.environ["IDA_LLM_API_KEY"],
        output_path=args.output,
        save_every=args.save_every,
    )

    generate_latex_case_table(args.output)

