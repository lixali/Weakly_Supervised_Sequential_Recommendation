# Weakly Supervised Domain Adaptation for Large Language Model Based Recommendation Systems

Large Language Models (LLMs) have achieved impressive results across a wide range of tasks, prompting significant interest in their application to recommendation systems. However, the significant domain gap between the web-based pretraining corpora of LLMs and the recommendation data leads to suboptimal performance of LLM-based models in recommendation tasks. This issue is becoming worse when target recommendation datasets are limited in size and highly sparse, because they are insufficient to finetune the recommendation model to close the domain gap.

To overcome the limited size and data sparsity challenge, we propose a new framework to construct a weakly supervised training dataset for LLMs to better adapt from the web domain to recommendation tasks. First, we curate recommendation-related documents data from web corpora in the **Document Filtering** stage. Second, we pair the curated documents with outlink documents to construct the training dataset in the **Behavior Linking** stage. Finally, we **weakly supervised train** the recommendation model using the collected dataset followed by the standard finetuning using the target recommendation data.

In this work, we utilize the ClueWeb for dataset construction and leverage the state-of-the-art LLM-based recommendation model HLLM as our backbone. Our experimental results on PixelRec200K (5% users) and Microlens-100K (15% users) datasets (limited size and highly sparse) demonstrate the efficacy of our proposed framework in this scenario.

## Quick highlights
- Two LLMs: item-level text encoder and user-level sequence encoder.
- Contrastive NCE training for ranking and retrieval.
- Works with pre-trained LLM weights (e.g., TinyLlama, Baichuan2) and ID-based baselines (HSTU, SASRec).
- Uses Deepspeed for memory-efficient and distributed training.

## Installation
Prerequisites:
- Python 3.10 (recommended for compatibility with torch 2.1.0)
- Git LFS if you plan to download large pre-trained weights

Quick setup:
```bash
conda create -n hllm python=3.10 -y
conda activate hllm
pip install -r requirements.txt
sudo apt update && sudo apt install git-lfs  # optional
```

Key packages (examples):
```
pytorch==2.1.0
deepspeed==0.14.2
transformers==4.41.1
lightning==2.4.0
flash-attn==2.5.9post1
```

## Data layout
Place processed interaction files under `dataset/` and item textual information under `information/` (these map to CLI `data_path` and `text_path` respectively):

```
dataset/
  ├─ amazon_books.csv
  ├─ Pixel1M.csv
  └─ Pixel8M.csv

information/
  ├─ amazon_books.csv
  ├─ Pixel1M.csv
  └─ Pixel8M.csv
```

See `code/` for utilities to process PixelRec and Books datasets.

## Training (example)
Set distributed env vars (`MASTER_ADDR`, `MASTER_PORT`, `NPROC_PER_NODE`, etc.) for multi-node runs. Model and hyper-parameters are controlled by YAML configs in `overall/`, `HLLM/`, and `IDNet/`, and by CLI args in `code/REC/utils/argument_list.py`.

Example (deepspeed):
```bash
python3 main.py \
  --config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
  --loss nce \
  --epochs 5 \
  --dataset Pixel200K \
  --train_batch_size 16 \
  --MAX_TEXT_LENGTH 256 \
  --MAX_ITEM_LIST_LENGTH 10 \
  --checkpoint_dir ./saved_path \
  --optim_args.learning_rate 1e-4 \
  --item_pretrain_dir /path/to/item_llm \
  --user_pretrain_dir /path/to/user_llm \
  --text_path /absolute/path/to/information \
  --text_keys '["title", "description"]'
```

Use `--gradient_checkpointing True` and Deepspeed stage 3 to save memory.

To run ID-based baselines, use `overall/ID.yaml` and the appropriate `IDNet/*.yaml` config.

## Evaluation / Inference
To evaluate a checkpoint, run the same command as training with `--val_only True` and point `--checkpoint_dir` to saved weights.

Pretrained fine-tuned weights are referenced in the original project; ensure you comply with third-party licenses when using them.

## Reproduce experiments
Reproduction scripts are in the `reproduce/` folder and cover Pixel8M and Books setups used in the paper.

## Acknowledgements
Thanks to RecBole, PixelRec, HSTU and other repositories whose code and ideas influenced this work. This repository is released under Apache License 2.0; verify third-party weights/licenses before use.

## Design
The proposed framework diagram is available below and in `design/proposed.pdf`.

<object data="design/proposed.pdf" type="application/pdf" width="100%" height="600">
  <p>Unable to display PDF inline. Download the diagram: <a href="design/proposed.pdf">design/proposed.pdf</a></p>
</object>
