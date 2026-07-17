# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.
# import ipdb; ipdb.set_trace()
from cProfile import run
from logging import getLogger
import torch
import json
from REC.data import *
from REC.config import Config
from REC.utils import init_logger, get_model, init_seed, set_color
from REC.trainer import Trainer
import torch.distributed as dist

import os
import numpy as np
import argparse
import torch.distributed as dist
import torch
import copy

# import traceback
# import pdb
# pdb.set_trace()


def convert_str(s):
    try:
        if s.lower() == 'none':
            return None
        if s.lower() == 'true':
            return True
        if s.lower() == 'false':
            return False
        float_val = float(s)
        if float_val.is_integer():
            return int(float_val)
        return float_val
    except ValueError:
        print(f"Unable to convert the string '{s}' to None / Bool / Float / Int, retaining the original string.")
        return s


def run_loop(local_rank, config_file=None, saved=True, extra_args=[]):

    # configurations initialization
    config = Config(config_file_list=config_file)
    config_for_eval = Config(config_file_list=config_file)
    
    device = torch.device("cuda", local_rank)
    config['device'] = device
    if len(extra_args):
        for i in range(0, len(extra_args), 2):
            key = extra_args[i][2:]
            value = extra_args[i + 1]
            try:
                if '[' in value or '{' in value:
                    value = json.loads(value)
                    if isinstance(value, dict):
                        for k, v in value.items():
                            value[k] = convert_str(v)
                    else:
                        value = [convert_str(x) for x in value]
                else:
                    value = convert_str(value)
                if '.' in key:
                    k1, k2 = key.split('.')
                    config[k1][k2] = value
                    config_for_eval[k1][k2] = value
                else:
                    config[key] = value
                    config_for_eval[key] = value
            except:
                raise ValueError(f"{key} {value} invalid")
    
    init_logger(config)
    init_logger(config_for_eval)
    logger = getLogger()    
    init_seed(config['seed'], config['reproducibility'])

    
    #### this if condition and rewriting of config_for_eval['dataset'] has to be here
    #### because line 54 if len(extra_args) will write the config_for_eval value into each key in the dictionary
    if config_for_eval.get('dataset_for_eval'): 
        
        logger.info(f"there is a dataset_for_eval, and it is {config_for_eval['dataset_for_eval']}")
        config_for_eval['dataset'] = config_for_eval['dataset_for_eval']
        logger.info(f"there is a dataset, after copying from dataset_for_eval and it is {config_for_eval['dataset']}")
    

    if 'text_path' in config:
        config['text_path'] = os.path.join(config['text_path'], config['dataset'] + '.csv')
        logger.info(f"Update text_path to {config['text_path']}")
    if 'text_path' in config_for_eval:
        config_for_eval['text_path'] = os.path.join(config_for_eval['text_path'], config_for_eval['dataset'] + '.csv')
        logger.info(f"Update text_path to {config_for_eval['text_path']}")

    # get model and data
    #### load_data return a Data class type (Data class is a self defined class) ; so dataload is a self defined Data class
    dataload = load_data(config, use_finetune_data_as_guidance=False)
    # breakpoint()
    train_loader, valid_loader, test_loader = bulid_dataloader(config, dataload)
    print(f"{len(train_loader) = }")
    
    if config["clueweb_pretrain"]: 
        dataload_for_eval = load_data(config_for_eval, use_finetune_data_as_guidance=True)
        # dataload_for_eval = load_data_for_eval(config)
        _, val_loader_for_eval, test_loader_for_eval = bulid_dataloader(config_for_eval, dataload_for_eval)

    model = get_model(config['model'])(config, dataload)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    world_size = torch.distributed.get_world_size()
    trainer = Trainer(config, model)
    logger.info(set_color('\nWorld_Size', 'pink') + f' = {world_size} \n')
    logger.info(config)
    logger.info(dataload)
    logger.info(model)
    # breakpoint()
    ### val_only , finetune_clueweb , gen_relevance_score , baseline_train , clueweb_pretrain can only have one True , rest are False
    if config['val_only'] or config["gen_relevance_score"]:
        ckpt_path = os.path.join(config['checkpoint_dir'], 'pytorch_model.bin')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        logger.info(f'Eval only model load from {ckpt_path}')
        msg = trainer.model.load_state_dict(ckpt, False)
        logger.info(f'{msg.unexpected_keys = }')
        logger.info(f'{msg.missing_keys = }')

        test_result = trainer.evaluate(test_loader, load_best_model=False, show_progress=config['show_progress'], init_model=True)
        
        logger.info(set_color('test result', 'yellow') + f': {test_result}')
    
    elif config['finetune_clueweb']:
        ckpt_path = os.path.join(config['checkpoint_dir'], 'pytorch_model.bin')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        logger.info(f'trying to finetune on this checkpoint {ckpt_path}')
        msg = trainer.model.load_state_dict(ckpt, False)
        logger.info(f'{msg.unexpected_keys = }')
        logger.info(f'{msg.missing_keys = }')
    
    if config["finetune_clueweb"] or config["baseline_train"] or config["clueweb_pretrain"]:
        # training process


        # if config["finetune_clueweb"] or config["baseline_train"] or config["clueweb_pretrain"]:
        if config["finetune_clueweb"] or config["baseline_train"]:
            best_valid_score, best_valid_result = trainer.fit(
                train_loader, valid_loader, saved=saved, show_progress=config['show_progress']
            )
        elif config["clueweb_pretrain"]:
            best_valid_score, best_valid_result = trainer.fit(
                train_loader, valid_loader, saved=saved, show_progress=config['show_progress']
            )
            # pass
            logger.info(f'Training Ended ' + set_color('best valid ', 'yellow') + f': {best_valid_result}')

        # model evaluation
        # if config["finetune_clueweb"] or config["baseline_train"] or config["clueweb_pretrain"]:
        if config["finetune_clueweb"] or config["baseline_train"]:
            test_result = trainer.evaluate(test_loader, load_best_model=saved, show_progress=config['show_progress'])

        elif config["clueweb_pretrain"]:
            test_result = trainer.evaluate(test_loader, load_best_model=saved, show_progress=config['show_progress'])

        logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
        logger.info(set_color('test result', 'yellow') + f': {test_result}')

        return {
            'best_valid_score': best_valid_score,
            'valid_score_bigger': config['valid_metric_bigger'],
            'best_valid_result': best_valid_result,
            'test_result': test_result
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", nargs='+', type=str)
    args, extra_args = parser.parse_known_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    config_file = args.config_file

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    try:
        run_loop(local_rank=local_rank, config_file=config_file, extra_args=extra_args)
    except Exception:
        if os.environ.get("POST_MORTEM_DEBUG", "False").lower() in {"1", "true", "yes"}:
            import traceback, pdb
            traceback.print_exc()
            pdb.post_mortem()
        raise
