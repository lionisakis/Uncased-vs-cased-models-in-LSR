import os
import json
import torch
import hydra
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import DictConfig
from transformers import AutoTokenizer

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .datasets.dataloaders import CollectionDataLoader
from .datasets.datasets import CollectionDatasetPreLoad
from .models.models_utils import get_model
from .tasks.transformer_evaluator import SparseIndexing
from .utils.utils import get_initialize_config
import math


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


def compute_confusion_matrix(loader, model, tokenizer, casing_map, device, is_query=True):
    """
    Compute the 2x2 casing confusion matrix.
    """
    cm = np.zeros((2, 2), dtype=int)

    for batch in tqdm(loader, desc="Processing batches"):
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # get per-token logits before aggregation
        # the encode() call inside Splade does log(1+relu()) and aggregation
        # but here we want *token-level logits* → use encode_ directly
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


    return cm


def plot_confusion_matrix(confusion_dict, out_path, labels=["Cased", "Uncased"], main_title="Casing Confusion Matrices"):
    """
    Plot multiple confusion matrices in a single figure with 2 columns and dynamic rows.
    
    Args:
        confusion_dict (dict): {name: confusion_matrix (2x2 numpy array)}
        out_path (str): path to save the figure
        labels (list): labels for x and y axes
        main_title (str): overall title for the figure
    """
    num_matrices = len(confusion_dict)
    ncols = 2
    nrows = math.ceil(num_matrices / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    axes = np.array(axes).reshape(-1)  # flatten in case of 1 row/col

    for idx, (name, cm) in enumerate(confusion_dict.items()):
        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues")

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # overlay numbers
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black", fontsize=12, fontweight="bold")

        ax.set_xlabel("Output Casing (Sparse Representation)")
        ax.set_ylabel("Input Casing (Tokenized Text)")
        ax.set_title(f"{name}", fontsize=14)

    # Hide any unused subplots
    for i in range(idx + 1, nrows * ncols):
        fig.delaxes(axes[i])

    fig.suptitle(main_title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)



@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def confusion_matrix(exp_dict: DictConfig):
    # Init configs
    exp_dict, config, init_dict, model_training_config = get_initialize_config(exp_dict)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_training_config["tokenizer_type"],
        use_fast=True
    )
    lowercase = config["lowercase"]

    # Build casing map
    casing_map = build_casing_map(tokenizer.vocab_size, tokenizer, lowercase)

    # Model
    model = get_model(config, init_dict).to(device)
    model.eval()

    # Queries/documents dataset
    q_collection = CollectionDatasetPreLoad(
        data_dir=exp_dict["data"]["flops_queries"],
        id_style="row_id"
    )
    q_loader = CollectionDataLoader(
        dataset=q_collection,
        tokenizer_type=model_training_config["tokenizer_type"],
        lowercase=lowercase,
        max_length=model_training_config["max_length"],
        batch_size=config["index_retrieve_batch_size"],
        shuffle=False,
        num_workers=1
    )

    print("ENCODE TEXTS INTO SPARSE REPRESENTATION")
    # confusion = compute_confusion_matrix(q_loader, model, tokenizer, casing_map, device)

    # Save JSON
    out_dir = exp_dict.config.out_dir
    # os.makedirs(out_dir, exist_ok=True)
    # json_path = os.path.join(out_dir, "confusion_matrix.json")
    # with open(json_path, "w") as f:
    #     json.dump({"confusion_matrix": confusion.tolist()}, f)
    # print(f"Confusion matrix saved to {json_path}")

    # Save PNG
    img_path = os.path.join(out_dir, "confusion_matrix.png")
    dictionary_of_confusion_matrix = {
        "Bert Cased no-preprocessing no-postprocessing": np.array([[3553950, 14216134],
                                                  [278552157, 11844734578]])
    }
    
    plot_confusion_matrix(dictionary_of_confusion_matrix, img_path)
    print(f"Confusion matrix image saved to {img_path}")


if __name__ == "__main__":
    confusion_matrix()
