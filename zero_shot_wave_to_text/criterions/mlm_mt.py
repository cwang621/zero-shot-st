# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.data.data_utils import post_process

from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss


@register_criterion("mlm_mt")
class MLMMT(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing, mlm_weight=1.0):
        super().__init__(task, sentence_avg, label_smoothing)
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.report_accuracy = True

        self.mlm_weight = mlm_weight


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument(
            "--mlm-weight",
            default=1.0,
            type=float,
            help="loss weight for masked language model task"
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if not model.training:
            src_tokens, src_lengths, prev_output_tokens = sample["net_input"].values()
            encoder_out = model.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
            net_output = model.decoder(
                prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
            )

            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
            logging_output = {
                "loss": utils.item(loss.data) if reduce else loss.data,
                "trans_loss": utils.item(loss.data) if reduce else loss.data,
                "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
            if self.report_accuracy:
                n_correct, total = self.compute_accuracy(model, net_output, sample)
                logging_output["n_correct"] = utils.item(n_correct.data)
                logging_output["total"] = utils.item(total.data)
        else:
            src_tokens, src_lengths, prev_output_tokens= sample["net_input"].values()
            encoder_out = model.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
            net_output = model.decoder(
                prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
            )
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
            logging_output = {
                "trans_loss": utils.item(loss.data) if reduce else loss.data,
                "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
            if self.mlm_weight > 0.0:
                masked_source = sample["masked_source"]
                mlm_target = sample["mlm_target"]
                mlm_encoder_out = model.encoder(
                    src_tokens=masked_source, src_lengths=src_lengths
                )
                mlm_loss = self.compute_mlm_loss(model, mlm_encoder_out, mlm_target)
                loss = loss + self.mlm_weight * mlm_loss
                logging_output["mlm_loss"] = utils.item(mlm_loss.data) if reduce else mlm_loss.data
                logging_output["mlm_ntokens"] = (mlm_target!=-100).sum().item()
            logging_output["loss"] = utils.item(loss.data) if reduce else loss.data
            if self.report_accuracy:
                n_correct, total = self.compute_accuracy(model, net_output, sample)
                logging_output["n_correct"] = utils.item(n_correct.data)
                logging_output["total"] = utils.item(total.data)

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, key="target", reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = sample[key]
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_mlm_loss(self, model, encoder_out, target):
        logits = model.encoder.compute_output_logits(encoder_out) # T x B x V
        logits = logits.transpose(0,1)
        V = logits.size(-1)
        loss = F.cross_entropy(
            logits.contiguous().view(-1, V), target.view(-1), reduction="sum"
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        trans_loss_sum = utils.item(
            sum(log.get("trans_loss", 0) for log in logging_outputs)
        )
        nll_loss_sum = utils.item(
            sum(log.get("nll_loss", 0) for log in logging_outputs)
        )
        mlm_loss_sum = utils.item(
            sum(log.get("mlm_loss", 0) for log in logging_outputs)
        )
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        mlm_ntokens = utils.item(sum(log.get("mlm_ntokens", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "trans_loss", trans_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        if mlm_ntokens > 0:
            metrics.log_scalar(
                "mlm_loss", mlm_loss_sum / mlm_ntokens / math.log(2), mlm_ntokens, round=3,
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
