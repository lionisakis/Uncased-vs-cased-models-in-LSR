import os
import json
import torch
import hydra
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import DictConfig
from transformers import AutoTokenizer
import math
from pathlib import Path

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .datasets.dataloaders import CollectionDataLoader
from .datasets.datasets import CollectionDatasetPreLoad
from .models.models_utils import get_model
from .tasks.transformer_evaluator import SparseIndexing
from .utils.utils import get_initialize_config


def build_casing_map(vocab_size, tokenizer, lowercase):
    """
    Precompute a mapping from vocab indices to cased/uncased categories.
    0 = cased, 1 = uncased
    """
    casing_map = np.ones(vocab_size, dtype=int)
    for i in range(vocab_size):
        token = tokenizer.convert_ids_to_tokens(i)
        if token is None or token in tokenizer.all_special_tokens:
            continue
        if lowercase:
            token = token.lower()
        casing_map[i] = 0 if any(c.isupper() for c in token) else 1
    return casing_map


def compute_confusion_matrix(loader, model, tokenizer, casing_map, device, is_query=True, max_print=0):
    """
    Compute the 2x2 casing confusion matrix.
    Optionally collect and print some unique cased OUTPUT tokens if max_print > 0.
    """
    cm = np.zeros((2, 2), dtype=int)
    collected_tokens = set()   # store unique cased OUTPUT tokens

    for batch in tqdm(loader, desc="Processing batches"):
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            logits = model.encode_tokens(batch, is_q=True)  # (bs, seq_len, vocab_size)

        input_ids = batch["input_ids"].cpu().numpy()
        
        for b in range(logits.shape[0]):
            for i, tok_id in enumerate(input_ids[b]):
                if tok_id in tokenizer.all_special_ids:
                    continue
        
                input_type = casing_map[tok_id]  # 0=cased, 1=uncased
                sparse_vec = logits[b, i].cpu().numpy()
                activated_ids = np.where(sparse_vec > 0)[0]
        
                cased_count = np.sum(casing_map[activated_ids] == 0)
                uncased_count = np.sum(casing_map[activated_ids] == 1)
        
                cm[input_type, 0] += cased_count
                cm[input_type, 1] += uncased_count

                # Collect unique cased OUTPUT tokens
                if max_print > 0:
                    for out_id in activated_ids:
                        if casing_map[out_id] == 0:  # only cased outputs
                            token = tokenizer.convert_ids_to_tokens(int(out_id))
                            if token not in tokenizer.all_special_tokens:
                                collected_tokens.add(token)

    # Print after full pass
    if max_print > 0 and collected_tokens:
        print(f"\nUnique cased OUTPUT tokens found: {len(collected_tokens)}")
        print(f"Showing up to {max_print}:")
        for tok in list(collected_tokens)[:max_print]:
            print(f"  {tok}")

    return cm


def plot_confusion_matrix(confusion_dict, out_path, labels=["Cased", "Uncased"], main_title="Casing Confusion Matrices"):
    num_matrices = len(confusion_dict)
    ncols = 2
    nrows = math.ceil(num_matrices / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 5))
    axes = np.array(axes).reshape(-1)  # flatten axes

    for idx, (name, cm) in enumerate(confusion_dict.items()):
        ax = axes[idx]

        # Use raw counts
        im = ax.imshow(cm, cmap="viridis")

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Overlay numbers with dynamic color based on background
        norm = plt.Normalize(vmin=cm.min(), vmax=cm.max())
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color_val = im.cmap(norm(cm[i, j]))  # returns RGBA tuple
                # compute luminance for color contrast
                luminance = 0.299*color_val[0] + 0.587*color_val[1] + 0.114*color_val[2]
                text_color = "white" if luminance < 0.5 else "black"
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=text_color, fontsize=12, fontweight="bold")

        ax.set_xlabel("Output Casing")
        ax.set_ylabel("Input Casing")
        ax.set_title(name, fontsize=12)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Count")

    # Hide unused axes
    for i in range(idx + 1, nrows * ncols):
        fig.delaxes(axes[i])

    fig.suptitle(main_title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, format=out_path.split(".")[-1])
    plt.close(fig)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def confusion_matrix(exp_dict: DictConfig):
    # Init configs
    exp_dict, config, init_dict, model_training_config = get_initialize_config(exp_dict)
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # # Tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_training_config["tokenizer_type"],
    #     use_fast=True
    # )
    # lowercase = config["lowercase"]

    # # Build casing map
    # casing_map = build_casing_map(tokenizer.vocab_size, tokenizer, lowercase)

    # # Model
    # model = get_model(config, init_dict).to(device)
    # model.eval()

    # # Queries/documents dataset
    # q_collection = CollectionDatasetPreLoad(
    #     data_dir=exp_dict["data"]["flops_queries"],
    #     id_style="row_id"
    # )
    # q_loader = CollectionDataLoader(
    #     dataset=q_collection,
    #     tokenizer_type=model_training_config["tokenizer_type"],
    #     lowercase=lowercase,
    #     max_length=model_training_config["max_length"],
    #     batch_size=config["index_retrieve_batch_size"],
    #     shuffle=False,
    #     num_workers=1
    # )

    # print("ENCODE TEXTS INTO SPARSE REPRESENTATION")
    # confusion = compute_confusion_matrix(q_loader, model, tokenizer, casing_map, device)

    # # Save JSON
    # out_dir = exp_dict.config.out_dir
    # os.makedirs(out_dir, exist_ok=True)
    # json_path = os.path.join(out_dir, "confusion_matrix.json")
    # with open(json_path, "w") as f:
    #     json.dump({"confusion_matrix": confusion.tolist()}, f)
    # print(f"Confusion matrix saved to {json_path}")

    # Save PNG

    Path("./figures").mkdir(parents=True, exist_ok=True)
    img_path = os.path.join("./figures", "confusion_matrix.png")
    dictionary_of_confusion_matrix = {
        "SPLADE-BERT Cased\nno-preprocessing no-postprocessing": np.array([[3553950, 14216134],
                                                  [278552157, 11844734578]]),
        "SPLADE-BERT Cased\nlowering no-postprocessing": np.array([[0, 0], [10917, 2131726362]]),
        "SPLADE-DistilBERT Cased\n no-preprocessing no-postprocessing": np.array([[1873723, 3650751], [116346609, 1127083068]]),
        "SPLADE-DistilBERT Cased\n lowering no-postprocessing": np.array([[0, 0], [1685, 1247906739]])
    }
    
    plot_confusion_matrix(dictionary_of_confusion_matrix, img_path)
    print(f"Confusion matrix image saved to {img_path}")


if __name__ == "__main__":
    confusion_matrix()
