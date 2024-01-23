# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math,os
from dataclasses import dataclass, field

import torch
from torch import nn
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import numpy as np 
from fairseq.distributed.utils import get_global_group, all_gather, get_world_size, get_rank
import torch.distributed as dist
import torch.nn.functional as F
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


@dataclass
class MultimodalCTLCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


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
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "multimodal_ctl_loss", dataclass=MultimodalCTLCriterionConfig
)
class MultimodalCTLCriterion(FairseqCriterion):
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
        self.padding_idx = 1 
        # self.temperature = 0.07
        self.args = task.args 
        self.encoder_bos_token = 0
        self.loss_type = getattr(self.args, 'loss_type', 'sim') 
        self.world_size = get_world_size(get_global_group())
        self.global_group = get_global_group()
        self.vit_type = getattr(self.args, 'ViT_type', 'clip')
        self.enable_addmargin = False 

        self.it_logit_scale = nn.Parameter(torch.ones(1) * np.log(1 / 0.07))
        self.tt_logit_scale = np.log(1 / 0.07)

    def gather_features(self, encoder_out_xlmr, encoder_out_clip_image, encoder_out_clip_text):
        bsz, fdim = encoder_out_xlmr.size()
        bsz = torch.LongTensor([bsz]).to(torch.cuda.current_device())
        gather_bsz = all_gather(bsz, self.global_group, return_tensor = True)
        max_bsz = gather_bsz.max() 
        dif_pad = max_bsz - gather_bsz[torch.cuda.current_device()]

        if dif_pad > 0:
            padding = torch.zeros(dif_pad, fdim).type_as(encoder_out_xlmr)
            padding = padding.detach()
            encoder_out_xlmr = torch.cat((encoder_out_xlmr, padding), dim=0)
            encoder_out_clip_image = torch.cat((encoder_out_clip_image, padding), dim=0)
            encoder_out_clip_text = torch.cat((encoder_out_clip_text, padding), dim=0)

        gather_xlmr = torch.cat(GatherFunc.apply(encoder_out_xlmr), dim = 0)  
        gather_image = torch.cat(GatherFunc.apply(encoder_out_clip_image), dim = 0)
        gather_ctext = torch.cat(GatherFunc.apply(encoder_out_clip_text), dim = 0)  

        if max_bsz > gather_bsz.float().mean():
            nonzero_order = gather_xlmr.sum(dim=-1).nonzero().squeeze(dim=-1)
            gather_xlmr = gather_xlmr.index_select(0, nonzero_order)
            gather_image = gather_image.index_select(0, nonzero_order)
            gather_ctext = gather_ctext.index_select(0, nonzero_order)
        
        return gather_xlmr, gather_image, gather_ctext

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        tgt_lang_id = int(sample["net_input"]['tgt_lang_id'][0])
        features = sample["net_input"]['features'] if 'features' in sample["net_input"] else None 
        _, encoder_out_clip_image = model.encoder.forward_image_clip(features)

        hyp_src_tokens = self.swap_tokens(sample['target'], append_bos_token = self.encoder_bos_token, left_pad = True)
        _, encoder_out_xlmr = model.encoder.forward_xlmr_text(hyp_src_tokens)  ### T x B x D  

        if tgt_lang_id == 1 and 'clip' in self.vit_type and 'nott' not in self.loss_type:
            encoder_out_clip_text, _ = model.encoder.forward_text_clip(hyp_src_tokens) 
        else:
            encoder_out_clip_text = encoder_out_clip_image ## compatible with the distributed training framework

        encoder_out_xlmr = F.normalize(encoder_out_xlmr, dim = -1) 
        encoder_out_clip_image = F.normalize(encoder_out_clip_image, dim = -1) 
        encoder_out_clip_text = F.normalize(encoder_out_clip_text, dim = -1) 

        if self.world_size == 1 or 'vanilla' in self.loss_type or not self.training:
            gather_xlmr = encoder_out_xlmr
            gather_image = encoder_out_clip_image
            gather_ctext = encoder_out_clip_text
        else:
            gather_xlmr, gather_image, gather_ctext = self.gather_features(encoder_out_xlmr, encoder_out_clip_image, encoder_out_clip_text)

        logit_scale = torch.clamp(self.it_logit_scale.exp(), max=100) 
        if 'noit' not in self.loss_type:
            it_ctl_loss = self.compute_ctl_loss(gather_xlmr, gather_image, logit_scale) * sample["ntokens"]
            loss = it_ctl_loss 
        else:
            loss = 0 
            it_ctl_loss = None
        
        if 'nott' in self.loss_type and 'clip' in self.vit_type :
            tt_ctl_loss = None 
        elif 'noit' in self.loss_type:
            tt_ctl_loss =  self.compute_ctl_loss(gather_xlmr, gather_ctext, self.tt_logit_scale, enable_addmargin = self.enable_addmargin) * sample["ntokens"] 
            loss =  tt_ctl_loss
        elif 'clip' in self.vit_type :
            tt_ctl_loss = 0.1 * self.compute_ctl_loss(gather_xlmr, gather_ctext, self.tt_logit_scale, enable_addmargin = self.enable_addmargin) * sample["ntokens"] 
            loss = loss + tt_ctl_loss 
        else:
            tt_ctl_loss = None 

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            "loss": loss.data,
            "it_ctl_loss": it_ctl_loss.data if it_ctl_loss is not None else 0,
            "tt_ctl_loss":  tt_ctl_loss.data if tt_ctl_loss is not None else 0,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

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


    def compute_ctl_loss(self, enc1, enc2, logit_scale, enable_addmargin=False):
        enc1, enc2 = enc1.float(), enc2.float()
        bsz, fdim = enc1.size()
        target = torch.arange(bsz).long().to(enc1.device)

        sim_i_2_t = logit_scale * enc1 @ enc2.t()
        sim_t_2_i = logit_scale * enc2 @ enc1.t()

        if enable_addmargin:
            sim_i_2_t = sim_i_2_t - 0.3 * torch.eye(bsz).type_as(sim_i_2_t)
            sim_t_2_i = sim_t_2_i - 0.3 * torch.eye(bsz).type_as(sim_t_2_i)

        loss_t_2_i = F.cross_entropy(sim_t_2_i, target)
        loss_i_2_t = F.cross_entropy(sim_i_2_t, target)
        loss = (loss_t_2_i + loss_i_2_t) / 2.0
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
        it_ctl_loss_sum = sum(log.get("it_ctl_loss", 0) for log in logging_outputs)
        tt_ctl_loss_sum = sum(log.get("tt_ctl_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "it_ctl_loss", it_ctl_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "tt_ctl_loss", tt_ctl_loss_sum / ntokens / math.log(2), ntokens, round=3
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
