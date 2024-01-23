#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import math
import os,random
import sys,faiss
from argparse import Namespace
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from omegaconf import DictConfig

def similarity_search(x, y, dim, normalize=False):

    num = x.shape[0]
    idx = faiss.IndexFlatL2(dim)
    if normalize:
        faiss.normalize_L2(x)
        faiss.normalize_L2(y)
    idx.add(x)
    scores, prediction = idx.search(y, 100)
    return prediction


def swap_tokens(target, append_bos_token = 0, padding_idx = 1):
    '''
    the CTL loss related to XLM-R only occurs for image-text pair and En-X pairs
    Thus, the src tokens length is small, or should obey the 77 token length limit for CLIP Text encoder.
    '''

    tgt_lengths = torch.LongTensor([s.ne(padding_idx).long().sum() for s in target])
    max_src_lengths = tgt_lengths.max() + 1   ## include bos token
    append_bos_token = torch.Tensor([append_bos_token]).type_as(target)
    hyp_src_tokens = torch.ones(tgt_lengths.size()[0], max_src_lengths).type_as(target)
    for idx, tgt_length in enumerate(tgt_lengths):
        src_token = torch.cat((append_bos_token, target[idx,:tgt_length]),-1)
        hyp_src_tokens[idx,-tgt_length-1:] = src_token
    return hyp_src_tokens


def calculate_i2t_retrieval(ifeature, tfeature, ids, image_ids, cal_i2t=True):  
    batch_size, dim = ifeature.shape 
    pred = similarity_search(ifeature, tfeature, dim, normalize=True)
    total = 0 
    r1, r5, r10 = 0, 0, 0
    
    for idx, sample_id in enumerate(ids):
        true_id = image_ids[sample_id]
        pred_image_ids = []

        for x in pred[idx]:
            item = image_ids[ids[x]]
            if item not in pred_image_ids:
                pred_image_ids.append(item)
            if len(pred_image_ids) > 10:
                break 

        if true_id == pred_image_ids[0]:
            r1 = r1 + 1
        
        if true_id in pred_image_ids[:5]:
            r5 = r5 + 1
        
        if true_id in pred_image_ids[:10]:
            r10 = r10 + 1 
        total = total + 1

    iname = 'i2t' if cal_i2t else 't2i'
    print(f"for {iname} task, R1 = \t {round(r1/total*100,1)}")
    print(f"for {iname} task, R5 = \t {round(r5/total*100,1)}")
    print(f"for {iname} task, R10= \t {round(r10/total*100,1)}")

    return round(r1/total*100,1), round(r5/total*100,1), round(r10/total*100,1)
    

def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    else:
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    # logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu
    # Load dataset splits
    task = tasks.setup_task(cfg.task, cfg_all = cfg) 


    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    # logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    if cfg.generation.lm_path is not None:
        overrides["data"] = cfg.task.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [cfg.generation.lm_path], arg_overrides=overrides, task=None
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({cfg.task.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": cfg.generation.lm_weight}
    generator = task.build_generator(
        models, cfg.generation, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    id_list = []    
    images_pooled_features, text_pooled_features, ctext_pooled_features = None, None, None 

    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        prefix_tokens = None
        if cfg.generation.prefix_size > 0:
            prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]

        hyp_src_tokens = swap_tokens(sample['target'])  
        with torch.no_grad():
            use_vanilla_clip = False 

            if use_vanilla_clip:
                _, cls_image_feature = models[0].encoder.clip.encode_image(sample['net_input']['features'].half()) 
                _, _, cls_text_feature = models[0].encoder.forward_text_clip(hyp_src_tokens)
                text_feature = cls_text_feature
            else:
                _, cls_image_feature = models[0].encoder.forward_image_clip(sample['net_input']['features'])  ## T x B x H
                if 'clip' in cfg.dataset.ViT_type and hasattr(models[0].encoder.clip, 'encode_text'):
                    cls_text_feature, _ = models[0].encoder.forward_text_clip(hyp_src_tokens)  ## T x B x H
                else:
                    cls_text_feature = None 
                
                _, text_feature = models[0].encoder.forward_xlmr_text(hyp_src_tokens)

        pool_image_feature = cls_image_feature
        pool_text_feature = text_feature
        pool_ctext_feature = cls_text_feature

        images_pooled_features = torch.cat((images_pooled_features,pool_image_feature), dim=0) if images_pooled_features is not None else pool_image_feature
        text_pooled_features = torch.cat((text_pooled_features,pool_text_feature), dim=0) if text_pooled_features is not None else pool_text_feature

        if 'clip' in cfg.dataset.ViT_type:
            ctext_pooled_features = torch.cat((ctext_pooled_features,pool_ctext_feature), dim=0) if ctext_pooled_features is not None else pool_ctext_feature
        else:
            ctext_pooled_features = None 
        id_list = id_list + [int(x) for x in sample['id']]

        torch.cuda.empty_cache()

    image_ids = [line.strip('\n') for line in open(f"{cfg.task.data}/test-ids.{cfg.task.target_lang}.raw.txt", 'r').readlines()]

    images_pooled_features = images_pooled_features.float().detach().cpu().numpy()
    text_pooled_features = text_pooled_features.float().detach().cpu().numpy()

    if 'clip' in cfg.dataset.ViT_type:
        ctext_pooled_features = ctext_pooled_features.float().detach().cpu().numpy()

    print(f">>> For {cfg.task.target_lang}, The i2t retrieval between clip img and xlmr text")
    ir1, ir5, ir10 = calculate_i2t_retrieval(images_pooled_features, text_pooled_features, id_list, image_ids)

    tr1, tr5, tr10 = calculate_i2t_retrieval(text_pooled_features, images_pooled_features,id_list, image_ids, cal_i2t = False)

    meanr = (ir1 + ir5 + ir10 + tr1 + tr5 + tr10) / 6.0 
    print(f">>> For {cfg.task.target_lang}, mean recall is {round(meanr,1)} ")


def cli_main():
    parser = options.get_generation_parser()
    parser.add_argument(
        '--arch', '-a', metavar='ARCH', default="wav2vec2",
        help='Model architecture. For constructing tasks that rely on '
             'model args (e.g. `AudioPretraining`)'
    )
    parser.add_argument('--pool-method', default='mean', help='tgt file')
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()

