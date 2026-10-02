# Amazon SASRec and HSTU baselines in Colab

These launchers train the ID-based baselines on the same Amazon Industrial and
Scientific subset used for the Gemma and TinyLlama experiments:

- `srun_baseline_amazon_industrial_and_scientific_sasrec.sh`
- `srun_baseline_amazon_industrial_and_scientific_hstu.sh`

The default dataset is
`amazon_industrial_and_scientific_9251_users_52568_interactions` (9,251 users,
25,848 items, and 52,568 interactions). Put the interaction CSV at:

```text
/content/Weekly_Supervised_Sequential_Recommendation/dataset/amazon_industrial_and_scientific_9251_users_52568_interactions.csv
```

Its required columns are `item_id,user_id,timestamp`. These ID-based models do not
need the item-information CSV or item-text embeddings.

Use a Colab GPU runtime. The large defaults are intended for an A100-class GPU;
an A100 with 80 GB offers more memory headroom. They are not guaranteed to fit a
T4. Both launchers check the GPU and prepare dependencies in `.venv_train`, reusing
the runtime's installed PyTorch and CUDA support. They do not require FlashAttention
or FBGEMM. Full Colab GPU training has not been validated locally.

## Model preparation

The default SASRec launcher uses the repository's paper-scale `LLMIDRec` variant:
a randomly initialized TinyLlama user encoder with trainable item-ID embeddings.
It downloads only TinyLlama's small architecture configuration, **not pretrained
language-model weights**. HSTU initializes its own weights and needs no model
download. Both models learn from the Amazon interactions.

Run each block below as a separate Colab cell. Keep `%%shell` on the first line so
the working directory and environment settings apply to the entire block and
output is streamed while the command runs. Do not prefix individual lines with
`!`.

First update the existing checkout and download the SASRec configuration:

```bash
%%shell
set -e
cd /content/Weekly_Supervised_Sequential_Recommendation
GIT_TERMINAL_PROMPT=0 timeout 60s git pull --ff-only
unset PYTHON_BIN PYTHON_EXEC
bash code/download_sasrec_config.sh
```

The configuration is saved to
`TinyLlama-1.1B-intermediate-step-1431k-3T/config.json` under the repository root.
If the TinyLlama directory is already present, it can be reused. Override
`PRETRAIN_DIR` for a different local configuration directory and use the same
value when launching SASRec. The SASRec launcher also downloads the configuration
automatically when needed.

## Run SASRec

```bash
%%shell
set -e
cd /content/Weekly_Supervised_Sequential_Recommendation/code
unset PYTHON_BIN PYTHON_EXEC
export DATASET=amazon_industrial_and_scientific_9251_users_52568_interactions
export GPU_IDS=0 NUM_GPUS=1 NPROC_PER_NODE=1
export BOOTSTRAP_TRAIN_ENV=True PYTHONUNBUFFERED=1
export TRAIN_BATCH_SIZE=64
bash srun_baseline_amazon_industrial_and_scientific_sasrec.sh
```

`SASREC_VARIANT=paper` is the default. To run the smaller conventional SASRec
implementation, add `export SASREC_VARIANT=standard` before the final command.
That variant uses two layers, four attention heads, and 512-dimensional embeddings
and does not need a model-configuration download. The two variants have different
architectures; record the selected variant with the reported result.

## Run HSTU

Run this cell after the SASRec cell has finished, or run it independently:

```bash
%%shell
set -e
cd /content/Weekly_Supervised_Sequential_Recommendation/code
unset PYTHON_BIN PYTHON_EXEC
export DATASET=amazon_industrial_and_scientific_9251_users_52568_interactions
export GPU_IDS=0 NUM_GPUS=1 NPROC_PER_NODE=1
export BOOTSTRAP_TRAIN_ENV=True PYTHONUNBUFFERED=1
export TRAIN_BATCH_SIZE=64
bash srun_baseline_amazon_industrial_and_scientific_hstu.sh
```

HSTU uses 22 layers, 32 heads, 2,048-dimensional item and hidden embeddings, and
linear dropout 0.5. These are the settings in the repository's existing `HSTU_1B`
recipes. With this Amazon item vocabulary and a maximum sequence length of 10,
the actual model has approximately **514 million parameters**.
The existing dense HSTU implementation does not apply its configured attention
dropout or relative attention bias; these launchers preserve that implementation.

For a smaller HSTU experiment, set these variables before the final command:

```bash
export HSTU_LAYERS=2 HSTU_HEADS=4 HSTU_DIM=512 HSTU_ITEM_DIM=512
```

This changes the experiment's model size. Keep the large defaults when reproducing
the existing large-model baseline settings.

## Preparation checks and common settings

To install dependencies and check the data, configuration, and GPU without starting
training, add this line before either launch command:

```bash
export PREPARE_ONLY=True
```

Run the original training cell without that line to start training. The launchers
print the selected dataset, model variant, interpreter, and checkpoint directory.

| Setting | Default |
| --- | --- |
| `DATASET` | `amazon_industrial_and_scientific_9251_users_52568_interactions` |
| `GPU_IDS`, `NPROC_PER_NODE` | `0`, `1` |
| `TRAIN_BATCH_SIZE` | `64` |
| `EVAL_BATCH_SIZE` | `128` |
| `NUM_WORKERS` | `2` |
| `MAX_ITEM_LIST_LENGTH` | `10` |
| `EPOCHS` | `201`, with validation-based early stopping |
| `STOPPING_STEP` | `10` (stop after 11 consecutive worse validation checks) |
| Learning rate, weight decay | `1e-3`, `0.1` |
| Loss, `NUM_NEGATIVES` | NCE, `0` (share the sampled negatives across the batch) |
| Distributed strategy | DeepSpeed ZeRO stage 2 |
| Precision | `bf16-mixed` |

The maximum sequence length is 10 to match the current Amazon HLLM launcher;
older PixelRec ID scripts used 50. Set `MAX_ITEM_LIST_LENGTH=50` if that is your
chosen comparison protocol. These launchers require `STAGE=2`, because the current
ID evaluation code does not support ZeRO stage 3 parameter gathering.

`NUM_NEGATIVES=0` preserves the existing baseline's shared-batch negative-sampling
path. Set `NUM_NEGATIVES=512` to request a fixed number instead; this changes the
negative-sampling setting and can increase memory use substantially.

For an out-of-memory error, first reduce `TRAIN_BATCH_SIZE`, for example to `8`.
This changes the effective batch size; keep track of it when comparing results.
Reducing HSTU's dimensions or using `SASREC_VARIANT=standard` also changes the model
being evaluated.

## Final test results

Training automatically evaluates the test set using the best saved checkpoint
after early stopping or after the final epoch. No separate testing command is
needed. The Colab output ends with a `FINAL TEST RESULTS` block containing Recall
and NDCG, and prints the location of `test_results.json` in the checkpoint
directory. For the default runs, these files are:

```text
model_amazon_industrial_and_scientific_9251_users_52568_interactions_sasrec_paper_baseline/test_results.json
model_amazon_industrial_and_scientific_9251_users_52568_interactions_hstu_baseline/test_results.json
```

Use the JSON file's `test_result` values for the paper. `best_valid_result` is
recorded separately for reference. These outputs are written only after a
successful test evaluation; a training or evaluation error still fails the run.
Updating the scripts does not retroactively create results for an earlier run.
