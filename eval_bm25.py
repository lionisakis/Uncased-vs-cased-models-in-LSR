import pyterrier as pt
import os
import sys

def eval_dataset(ds_name, index_path):
    print(f"=== BM25 on {ds_name} ===")
    ds = pt.get_dataset(ds_name)
    indexer = pt.IterDictIndexer(index_path, overwrite=False)

    if not os.path.exists(f"{index_path}/data.properties"):
        index_ref = indexer.index(ds.get_corpus_iter(), fields=["text"])
    else:
        index_ref = pt.IndexRef.of(f"{index_path}/data.properties")

    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")
    res = pt.Experiment(
        [bm25],
        ds.get_topics(),
        ds.get_qrels(),
        eval_metrics=["map", "ndcg_cut_10", "recall_100", "mrr"],
        names=["BM25"]
    )
    print(res)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_bm25.py <dataset-name> <index-path>")
        sys.exit(1)

    dataset = sys.argv[1]
    index_path = sys.argv[2] if len(sys.argv) > 2 else f"./index_{dataset.replace('/', '_')}"

    eval_dataset(dataset, index_path)
