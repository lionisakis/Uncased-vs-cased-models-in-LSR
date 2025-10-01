# Uncased vs. Cased Models in LSR

This repository provides code for **training**, **indexing**, and **retrieval** using SPLADE models. It also includes scripts to run evaluations on the [BEIR benchmark](https://github.com/beir-cellar/beir).

---

## 🚀 Getting Started

### Requirements

To set up the environment for **training** and **evaluation**, run:

```bash
conda env create -f environment.yaml
```

> ⚠️ BM25 requires a separate environment. Use:
```bash
conda env create -f bm25_environment.yaml
```

---

### Dataset

To simplify setup, we provide pre-processed data folders that can be downloaded here:  
📦 [SPLADE Data (queries, documents, and hard negatives)](https://download.europe.naverlabs.com/splade/sigir22/data.tar.gz)  
📦 [Vienna Triplets (for distillation)](https://www.dropbox.com/s/sl07yvse3rlowxg/vienna_triplets.tar.gz?dl=0)

Extract the archives in the repository root:

```bash
tar -xzvf data.tar.gz
tar -xzvf vienna_triplets.tar.gz
```

Rename the distillation folder as follows:

```bash
mv data/vienna_triplets data/distil_mse
```

---

## 🏋️ Train Models

To train models, run:

```bash
conda activate splade
export PYTHONPATH=$PYTHONPATH:$(pwd)
export SPLADE_CONFIG_NAME={experiment.yaml}
python3 -m splade.all
```

Alternatively, you can use the provided SLURM scripts in `slurm_jobs/train/`.

---

## 📊 Evaluate on BEIR

You can evaluate trained models on BEIR datasets with:

```bash
conda activate splade
export PYTHONPATH=$PYTHONPATH:$(pwd)
export SPLADE_CONFIG_FULLPATH={experiment.yaml}

for dataset in arguana fiqa nfcorpus quora scidocs scifact trec-covid webis-touche2020 climate-fever dbpedia-entity fever hotpotqa nq
do
    python3 -m splade.beir_eval
                +beir.dataset=$dataset
                +beir.dataset_path=data/beir
done
```

Or use the SLURM scripts in `slurm_jobs/beir/`.

---

## 📚 BM25 Evaluation

To run BM25 baselines:

```bash
conda activate bm25

echo "Running BM25 evaluations with PyTerrier..."

python eval_bm25.py
```

Or use the SLURM scripts in `slurm_jobs/bm25/`.

---

## 📂 Project Structure

```
.
├── slurm_jobs/           # SLURM job submission scripts
│   ├── train/            # Training jobs
│   ├── beir/             # BEIR evaluation jobs
│   └── bm25.job             # BM25 evaluation jobs
├── splade/               # Core SPLADE code
├── environment.yaml      # Conda environment for SPLADE
├── bm25_environment.yaml # Conda environment for BM25
└── eval_bm25.py          # BM25 evaluation script
```

---

## ℹ️ Additional Information

This repository is a fork of the [original SPLADE repository](https://github.com/naver/splade).  
For detailed documentation and additional usage instructions, please refer to the original repo.
