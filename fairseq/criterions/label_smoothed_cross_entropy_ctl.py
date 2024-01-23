# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch.nn as nn
import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import torch.nn.functional as F
from fairseq.distributed.utils import get_global_group, all_gather, get_world_size, get_rank
import torch.distributed as dist
import numpy as np 
import logging

logger = logging.getLogger(__name__)


class GatherFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output) 
    
    @staticmethod
    def backward(ctx, *grads):
        input,  = ctx.saved_tensors
        grads = torch.stack(grads, dim = 0)
        dist.all_reduce(grads)
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("label_smoothed_cross_entropy_ctl")
class LabelSmoothedCrossEntropyCtlCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.args = task.args if hasattr(task,'args') else None 
        self.temperature = getattr(self.args,'contrastive_temperature', 0.1)
        self.contrastive_lambda = getattr(self.args,'contrastive_lambda', 1.0)
        self.enable_cls_pooling = getattr(self.args,'enable_cls_pooling', False)

        self.bos_idx = 0 
        self.mask_idx = 250001 ## fixed for XLM-R model
        self.padding_idx = 1
        self.enable_gather = True 

        self.global_group = get_global_group()
        self.world_size = get_world_size(self.global_group)


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        parser.add_argument('--contrastive-lambda', default=3.0, type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--contrastive-temperature', default=0.1, type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--is-mono-ctl', action='store_true',
                            help='report accuracy metric')
        # fmt: on

    def forward(self, model, sample, reduce=True, epoch=1, *args, **kwargs):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss   fairseq/criterions/label_smoothed_cross_entropy.py:16
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"], epoch=epoch)
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        if self.enable_gather and self.world_size > 1:
            bsz = sample['id'].size(0)
            bsz = torch.LongTensor([bsz]).to(torch.cuda.current_device())
            gather_bsz = all_gather(bsz, self.global_group, return_tensor = True)
        else:
            gather_bsz = None 

        ctl_loss = self.compute_para_contrastive_loss(model, sample, net_output, gather_bsz = gather_bsz) * sample["ntokens"] / sample["target"].size(0)
        ctl_loss =  self.contrastive_lambda * ctl_loss 
        loss  = loss + ctl_loss

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ctl_loss": ctl_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "src_lang_id": int(sample["net_input"]["src_lang_id"][0].cpu()) if "src_lang_id" in sample["net_input"].keys() else None,
            "tgt_lang_id": int(sample["net_input"]["tgt_lang_id"][0].cpu()) if "tgt_lang_id" in sample["net_input"].keys() else None,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss
    
    def swap_tokens(self, target, append_bos_token = 0, left_pad = True):
        '''
        the CTL loss related to XLM-R only occurs for image-text pair and En-X pairs
        Thus, the src tokens length is small, or should obey the 77 token length limit for CLIP Text encoder.
        '''
        if not left_pad:
            append_bos_token = torch.Tensor([append_bos_token]).repeat(target.size(0), 1).type_as(target)
            hyp_src_tokens = torch.cat((append_bos_token, target), dim = -1)
        else:
            tgt_lengths = torch.LongTensor([s.ne(self.padding_idx).long().sum() for s in target])
            max_src_lengths = tgt_lengths.max() + 1   ## include bos token
            append_bos_token = torch.Tensor([append_bos_token]).type_as(target)
            hyp_src_tokens = torch.ones(tgt_lengths.size()[0], max_src_lengths).type_as(target)
            for idx, tgt_length in enumerate(tgt_lengths):
                src_token = torch.cat((append_bos_token, target[idx,:tgt_length]),-1)
                hyp_src_tokens[idx,-tgt_length-1:] = src_token
        return hyp_src_tokens


    def compute_para_contrastive_loss(self, model, sample, net_output, enable_addmargin=False, gather_bsz=None):
        src_tokens = sample["net_input"]['src_tokens']
        enc_src = net_output[1]['encoder_out'][0]
        enc_src = enc_src.transpose(0,1)

        src_mask = (src_tokens != self.padding_idx)
        enc_src = (enc_src * src_mask.unsqueeze(-1)).sum(dim=1) / src_mask.float().sum(dim=1).unsqueeze(-1)

        enc_src =  F.normalize(enc_src, dim = -1)
        _, fdim = enc_src.size()

        hyp_src_tokens = self.swap_tokens(sample['target'])
        enc_tgt = model.encoder.forward(hyp_src_tokens)['encoder_out'][0]
        enc_tgt = enc_tgt.transpose(0,1)

        tgt_mask = (hyp_src_tokens != self.padding_idx)
        enc_tgt = (enc_tgt * tgt_mask.unsqueeze(-1)).sum(dim=1) / tgt_mask.float().sum(dim=1).unsqueeze(-1)
        enc_tgt =  F.normalize(enc_tgt, dim = -1)

        if self.world_size == 1 or not self.training or gather_bsz is None:
            gather_src = enc_src
            gather_tgt = enc_tgt
        else:
            max_bsz = gather_bsz.max()
            dif_pad = max_bsz - gather_bsz[torch.cuda.current_device()]

            if dif_pad > 0:
                padding = torch.zeros(dif_pad, fdim).type_as(enc_src)
                padding = padding.detach()
                enc_src = torch.cat((enc_src, padding), dim=0)
                enc_tgt = torch.cat((enc_tgt, padding), dim=0)
            
            gather_src = torch.cat(GatherFunc.apply(enc_src), dim = 0)  
            gather_tgt = torch.cat(GatherFunc.apply(enc_tgt), dim = 0)  

            if max_bsz > gather_bsz.float().mean():
                nonzero_order = gather_src.sum(dim=-1).nonzero().squeeze(dim=-1)
                gather_src = gather_src.index_select(0, nonzero_order)
                gather_tgt = gather_tgt.index_select(0, nonzero_order)

        enc1, enc2 = gather_src.float(), gather_tgt.float()
        bsz, fdim = enc1.size()
        target = torch.arange(bsz).long().to(enc1.device) 

        sim_i2t = enc1 @ enc2.t() / self.temperature  
        sim_t2i = enc2 @ enc1.t() / self.temperature

        if enable_addmargin:
            sim_i2t = sim_i2t - 0.3 * torch.eye(bsz).type_as(sim_i2t)
            sim_t2i = sim_t2i - 0.3 * torch.eye(bsz).type_as(sim_t2i)
            
        loss_i2t = F.cross_entropy(sim_i2t, target)
        loss_t2i = F.cross_entropy(sim_t2i, target)

        loss = (loss_i2t + loss_t2i) / 2.0 

        return loss

        
    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ctl_loss_sum = sum(log.get("ctl_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "ctl_loss", ctl_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
