"""Fail before torchrun on malformed data, missing imports, or unsuitable CUDA."""
import csv
from collections import Counter
from pathlib import Path
import sys


def check_interactions(path):
    users = Counter()
    items = set()
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.reader(stream)
        if next(reader, None) != ['item_id', 'user_id', 'timestamp']:
            raise ValueError('CSV columns must be item_id,user_id,timestamp in that order.')
        for row_number, row in enumerate(reader, 2):
            if len(row) != 3 or not all(row):
                raise ValueError(f'Invalid or empty fields on CSV row {row_number}.')
            try:
                int(row[2])
            except ValueError as exc:
                raise ValueError(f'Timestamp on CSV row {row_number} must be an integer.') from exc
            items.add(row[0])
            users[row[1]] += 1
    if not users or min(users.values()) < 4:
        raise ValueError('Each user needs at least 4 interactions: 2 training, 1 validation, 1 test.')
    if len(items) < 200:
        raise ValueError('The configured Recall/NDCG@200 evaluation needs at least 200 items.')
    print(f'Data verified: {len(users):,} users, {len(items):,} items, '
          f'{sum(users.values()):,} interactions.', flush=True)


def main():
    path, model_config, processes, precision = sys.argv[1:]
    check_interactions(path)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable. Select a GPU runtime in Colab, then rerun.')
    if not 1 <= int(processes) <= torch.cuda.device_count():
        raise ValueError('NPROC_PER_NODE must fit the GPUs selected by GPU_IDS.')
    for index in range(int(processes)):
        with torch.cuda.device(index):
            if precision.startswith('bf16') and not torch.cuda.is_bf16_supported():
                raise RuntimeError('bf16 requires a supported GPU (for example A100). '
                                   'Select A100 or explicitly set PRECISION=16-mixed.')
            properties = torch.cuda.get_device_properties(index)
            print(f'GPU {index}: {properties.name}, '
                  f'{properties.total_memory / 2**30:.1f} GiB', flush=True)
    # Exercise the same eager imports as training before spawning a worker.
    import run
    from REC.config import Config
    config = Config(['overall/ID_deepspeed.yaml', model_config])
    if config['model'] == 'LLMIDRec':
        from REC.model.HLLM.modeling_llama import LlamaForCausalLM
    print(f'Training imports verified; model={config["model"]}.', flush=True)


if __name__ == '__main__':
    main()
