import pyterrier as pt
import os
import json
from ir_measures import MAP, NDCG, Recall, MRR
import pandas as pd

def get_max_docno_length(ds):
    max_len = 0
    for doc in ds.get_corpus_iter():
        docno_len = len(str(doc.get("docno", "")))
        if docno_len > max_len:
            max_len = docno_len
    # Add a small buffer
    return max_len + 5


def init_pt():
    if not pt.started():
        pt.init()

def eval_dataset(ds_name, index_path, metrics, out_path=None):
    print(f"\n=== BM25 on {ds_name} ===")
    ds = pt.get_dataset(ds_name)

    max_docno_len = get_max_docno_length(ds)

    # Ensure index folder exists
    os.makedirs(index_path, exist_ok=True)

    # Fix for docno length issue: set meta length higher
    indexer = pt.IterDictIndexer(index_path, overwrite=False, fields=("text",), meta={'docno': max_docno_len})

    if not os.path.exists(f"{index_path}/data.properties"):
        print("Index not found, creating...")
        index_ref = indexer.index(ds.get_corpus_iter())
    else:
        index_ref = pt.IndexRef.of(f"{index_path}/data.properties", verbose=True)

    topics = ds.get_topics()

    # Ensure query column exists
    if 'query' not in topics.columns:
        for col in ['text', 'description', 'narrative']:
            if col in topics.columns:
                topics['query'] = topics[col].astype(str)
                break
        else:
            raise ValueError(f"No usable query column found in {ds_name}. Available columns: {topics.columns}")

    # Clean queries
    import re

    def clean_query(q):
        if isinstance(q, list):  # flatten lists like authors/references
            q = " ".join(map(str, q))
        q = str(q)
        # q = re.sub(r'[:?"<>]', ' ', q)  # remove bad chars
        q = re.sub(r'[^0-9a-zA-Z\s]', ' ', q)
        return q.strip()

    topics['query'] = topics['query'].map(clean_query)
    topics = topics[topics['query'].str.strip() != ""]
    topics = topics[~pd.isna(topics['query'])]

    # Use safe query parsing
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25", controls={"parse.controls": "off"}, verbose=True)

    res = pt.Experiment(
        [bm25],
        topics,
        ds.get_qrels(),
        eval_metrics=metrics,
        names=[f"BM25 {ds_name}"],
        verbose=True
    )

    # Print full results
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print("Full results:")
        print(res)

    # Save results periodically if path provided
    if out_path:
        if os.path.exists(out_path):
            with open(out_path, "r") as f:
                existing_results = json.load(f)
        else:
            existing_results = {}

        existing_results[ds_name] = res.to_dict(orient="records")
        with open(out_path, "w") as f:
            json.dump(existing_results, f, indent=4)
        print(f"Results saved/updated for {ds_name} at {out_path}")

    return res

if __name__ == "__main__":
    out_path = "./models/bm25/out/results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    init_pt()

    msmarco_index= "./models/bm25/index/msmarco"
    
    # MSMARCO dev
    msmarco_metrics = [
        MRR@10,
        Recall@5, Recall@10, Recall@15, Recall@20, Recall@30, Recall@100, Recall@200, Recall@500, Recall@1000,
    ]
    eval_dataset("irds:msmarco-passage/dev",msmarco_index, msmarco_metrics, out_path)

    # TREC DL 2019
    trec_metrics = [
        MRR@10,
        Recall@5, Recall@10, Recall@15, Recall@20, Recall@30, Recall@100, Recall@200, Recall@500, Recall@1000,
        NDCG@5, NDCG@10, NDCG@15, NDCG@20, NDCG@30, NDCG@100, NDCG@200, NDCG@500, NDCG@1000
    ]

    eval_dataset("irds:msmarco-passage/trec-dl-2019", msmarco_index, trec_metrics, out_path)

    # TREC DL 2020
    eval_dataset("irds:msmarco-passage/trec-dl-2020", msmarco_index, trec_metrics, out_path)

    # BEIR datasets
    beir_datasets = [
        "arguana", 
        "climate-fever", 
        "fiqa/test",
        "nfcorpus/test", 
        "quora/test", 
        "trec-covid",
        "webis-touche2020", 
        "dbpedia-entity/test", 
        "fever/test", 
        "hotpotqa/test", 
        "nq",
        "scidocs",
        "scifact/test",  
    ]
    for ds in beir_datasets:
        eval_dataset(f"irds:beir/{ds}", f"./models/bm25/index/{ds}", trec_metrics, out_path)
