# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
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
import clip,os
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
class MultimodalCocoCTLCriterionConfig(FairseqDataclass):
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
    "multimodal_coco_ctl_loss", dataclass=MultimodalCocoCTLCriterionConfig
)
class MultimodalCocoCTLCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        '''
        In this loss, all samples with the same image-path are treated as the positive samples when calculating contrastive loss. 
        This is different from that in `multimodal_ctl_loss.py` where the positive sample only contains itself. 
        '''
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.padding_idx = 1 
        self.temperature = 0.07
        self.args = task.args 
        self.encoder_bos_token = 0
        self.loss_type = getattr(self.args, 'loss_type', 'sim') 
        self.world_size = get_world_size(get_global_group())
        self.global_group = get_global_group()

        self.it_logit_scale = nn.Parameter(torch.ones(1) * np.log(1 / 0.07))
        self.tt_logit_scale = np.log(1 / 0.01)


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        tgt_lang_id = int(sample["net_input"]['tgt_lang_id'][0])

        _, encoder_out_clip_image = model.encoder.forward_image_clip(sample["net_input"]['features']) 
        encoder_out_clip_image = encoder_out_clip_image.contiguous()
        hyp_src_tokens = self.swap_tokens(sample['target'], append_bos_token = self.encoder_bos_token, left_pad = True)
        _, encoder_out_xlmr = model.encoder.forward_xlmr_text(hyp_src_tokens)  ### T x B x D

        if tgt_lang_id == 1:
            encoder_out_clip_text, clip_text_tokens  = model.encoder.forward_text_clip(hyp_src_tokens) 
        else:
            encoder_out_clip_text = encoder_out_clip_image

        gather_xlmr = F.normalize(encoder_out_xlmr, dim = -1) 
        gather_image = F.normalize(encoder_out_clip_image, dim = -1) 
        gather_ctext = F.normalize(encoder_out_clip_text, dim = -1) 

        logit_scale = torch.clamp(self.it_logit_scale.exp(), max=100)
        it_ctl_loss = self.compute_coco_ctl_loss(gather_xlmr, gather_image, logit_scale, sample) * sample["ntokens"] 
        tt_ctl_loss = 0.1 * self.compute_coco_ctl_loss(gather_xlmr, gather_ctext, self.tt_logit_scale, sample) * sample["ntokens"] 
        loss = tt_ctl_loss + it_ctl_loss 

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "it_ctl_loss": it_ctl_loss.data,
            "tt_ctl_loss": tt_ctl_loss.data, 
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

    
    def compute_coco_ctl_loss(self, enc_src, enc_tgt, logit_scale, sample):
        image_ids = sample['image_ids']   ## positive samples if having the same image-ids
        bsz, _ = image_ids.size()
        _, fdim = enc_src.size()

        if self.world_size > 1: 
            bsz = torch.LongTensor([bsz]).to(torch.cuda.current_device())
            gather_bsz = all_gather(bsz, self.global_group, return_tensor = True)
            max_bsz = gather_bsz.max() 
            dif_pad = max_bsz - gather_bsz[torch.cuda.current_device()]

            if dif_pad > 0:
                padding = torch.zeros(dif_pad, fdim).type_as(enc_src)
                padding = padding.detach()
                enc_src = torch.cat((enc_src, padding), dim=0)
                enc_tgt = torch.cat((enc_tgt, padding), dim=0)

                pad_id = torch.zeros(dif_pad, 1).type_as(image_ids)
                image_ids = torch.cat((image_ids, pad_id), dim=0)

            gather_src = torch.cat(GatherFunc.apply(enc_src), dim = 0)  
            gather_tgt = torch.cat(GatherFunc.apply(enc_tgt), dim = 0)  
            gather_image_ids = torch.cat(GatherFunc.apply(image_ids), dim = 0)    

            if max_bsz > gather_bsz.float().mean():
                nonzero_order = gather_src.sum(dim=-1).nonzero().squeeze(dim=-1)
                gather_src = gather_src.index_select(0, nonzero_order)
                gather_tgt = gather_tgt.index_select(0, nonzero_order)
                gather_image_ids = gather_image_ids.index_select(0, nonzero_order) 
        else:
            gather_src = enc_src 
            gather_tgt = enc_tgt
            gather_image_ids = image_ids

        bsz, _ = gather_image_ids.size()
        r_image_ids = gather_image_ids.repeat(1,bsz)
        l_image_ids = gather_image_ids.t().repeat(bsz, 1)
        sim_targets = (r_image_ids == l_image_ids).byte()

        enc1, enc2 = gather_src.float(), gather_tgt.float()
        enc1, enc2 = enc1.float(), enc2.float()
        bsz, fdim = enc1.size()
        sim_i2t = logit_scale * enc1 @ enc2.t()
        sim_t2i = logit_scale * enc2 @ enc1.t()

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean()
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
