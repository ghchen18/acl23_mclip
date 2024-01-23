import ast
import logging
import math
import os,re
import sys
from argparse import Namespace
import numpy as np
import torch
from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf


def main(cfg):
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

def _main(cfg, output_file):
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
    task = tasks.setup_task(cfg)

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))

    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task, 
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    ### load ckpt from xlmrL ckpts
    from fairseq.models.roberta import XLMRModel
    workloc = cfg.common_eval.path.split('/models/')[0] #"/path/to/your/workloc"
    
    if 'xlmrL' in cfg.common_eval.path:
        xlmr_params = torch.load(f'{workloc}/models/xlmrL_base/model.pt',  map_location=torch.device("cpu"))
    else:
        xlmr_params = torch.load(f'{workloc}/models/xlmr_base/model.pt',  map_location=torch.device("cpu"))

    ## load ckpt from trained NMT models
    update_model_dict = {k.replace('model.', ''):v for k, v in models[0].encoder.xlmr.state_dict().items()}
    del update_model_dict['_float_tensor']
    save_dir = cfg.common_eval.path.replace('/checkpoint_best.pt', '')

    state_dict = {"model": update_model_dict, "args": xlmr_params['args']}
    torch.save(state_dict, f'{save_dir}/mte_mclip.pt')

def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
