import math
import torch
import logging
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.data.data_utils import post_process

from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss


logger = logging.getLogger(__name__)

@register_criterion("chimera_st")
class ChimeraST(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing, post_process="letter", mlm_weight=1.0,
                 align_weight=1.0, ctc_weight=1.0, memory_num=64, type="contrastive",contrastive_temp=0.1):
        super().__init__(task, sentence_avg, label_smoothing)
        self.blank_idx = task.source_dictionary.index(task.blank_symbol) if hasattr(task, 'blank_symbol') else 0
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.report_accuracy = True

        self.align_weight = align_weight
        self.mlm_weight = mlm_weight
        self.ctc_weight = ctc_weight

        self.memory_num = memory_num
        self.zero_infinity = True
        self.post_process = post_process
        self.type = type
        self.contrastive_temp = contrastive_temp


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument(
            "--post-process",
            default="letter",
            type=str,
            help="how to post process predictions into words. can be letter, "
                 "wordpiece, BPE symbols, etc. "
                 "See fairseq.data.data_utils.post_process() for full list of options",
        )
        parser.add_argument(
            "--mlm-weight",
            default=1.0,
            type=float,
            help="weight for mlm loss"
        )
        parser.add_argument(
            "--align-weight",
            default=1.0,
            type=float,
            help="weight for alignment loss"
        )
        parser.add_argument(
            "--ctc-weight",
            default=1.0,
            type=float,
            help="weight for ctc loss"
        )
        parser.add_argument(
            "--type",
            default="contrastive",
            type=str,
            help="choose from [contrastive, mse]"
        )
        parser.add_argument(
            "--contrastive-temp",
            default=0.1,
            type=float,
            help="temp for contrastive loss"
        )


    def forward(self, model, sample, reduce=True):
        net_input = sample["net_input"]
        src_tokens, src_lengths, prev_output_tokens = net_input["src_tokens"], net_input["src_lengths"], net_input["prev_output_tokens"]
        mode = net_input.get("mode", "speech")

        if mode == "text":
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
                "text_ntokens": sample["ntokens"],
                "text_nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
            if self.report_accuracy:
                n_correct, total = self.compute_accuracy(model, net_output, sample)
                logging_output["n_correct"] = utils.item(n_correct.data)
                logging_output["total"] = utils.item(total.data)
            if self.mlm_weight > 0.0:
                masked_source = sample["masked_source"]
                mlm_target = sample["mlm_target"]
                mlm_encoder_out = model.encoder(
                    src_tokens=masked_source, src_lengths=src_lengths, mode=mode
                )
                mlm_loss = self.compute_mlm_loss(model, mlm_encoder_out, mlm_target)
                loss += self.mlm_weight * mlm_loss
                logging_output["mlm_loss"] = utils.item(mlm_loss.data) if reduce else mlm_loss.data
                logging_output["mlm_ntokens"] = (mlm_target!=-100).sum().item()
            logging_output["loss"] = utils.item(loss.data) if reduce else loss.data
        elif mode == "speech":
            if not model.training:
                encoder_out = model.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
                net_output = model.decoder(
                    prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
                )
                loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
                sample_size = (
                    sample["ntokens"]
                )
                logging_output = {
                    "loss": utils.item(loss.data) if reduce else loss.data,
                    "trans_loss": utils.item(loss.data) if reduce else loss.data,
                    "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
                    "ntokens": sample["ntokens"],
                    "nsentences": sample["target"].size(0),
                    "text_ntokens": sample["ntokens"],
                    "sample_size": sample_size,
                }
                n_correct, total = self.compute_accuracy(model, net_output, sample)
                logging_output["n_correct"] = utils.item(n_correct.data)
                logging_output["total"] = utils.item(total.data)
            else:
                speech_encoder_out = model.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
                text_tokens, text_lengths, ntokens = sample["transcript"].values()
                with torch.no_grad():
                    text_encoder_out = model.encoder(src_tokens=text_tokens, src_lengths=text_lengths)
                ad_loss = self.compute_ad_loss(speech_encoder_out, text_encoder_out, reduce)
                sample_size = (
                    sample["ntokens"]
                )
                logging_output = {
                    "ad_loss": utils.item(ad_loss.data) if reduce else ad_loss.data,
                    "ntokens": ntokens,
                    "nsentences": sample["target"].size(0),
                    "spch_ntokens": ntokens,
                    "spch_nsentences": sample["target"].size(0),
                    "sample_size": sample_size,
                    "lengths": (sample["target"].size(0) * self.memory_num) if self.memory_num>0 else ntokens,
                }
                loss = self.align_weight * ad_loss
                if self.ctc_weight > 0.0:
                    ctc_loss = self.compute_ctc_loss(model, sample, speech_encoder_out)
                    loss += self.ctc_weight * ctc_loss
                    logging_output["ctc_loss"] = utils.item(ctc_loss.data) if reduce else ctc_loss.data
                logging_output["loss"] = utils.item(loss.data) if reduce else loss.data
        else:
            raise ValueError("only support text and speech mode")

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

    def compute_ctc_loss(self, model, sample, encoder_out):
        transcript = sample["transcript"]
        ctc_logit = model.encoder.compute_ctc_logits(encoder_out)
        ctc_logit = ctc_logit.contiguous() # T x B x C
        lprobs = model.get_normalized_probs(
            [ctc_logit], log_probs=True
        ).contiguous()  # (T, B, C) from the encoder
        non_padding_mask = ~encoder_out["memory_info"]["memory_padding_mask"]
        input_lengths = non_padding_mask.long().sum(-1)

        pad_mask = (transcript["tokens"] != self.pad_idx) & (
                transcript["tokens"] != self.eos_idx
        )
        targets_flat = transcript["tokens"].masked_select(pad_mask)
        transcript_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                transcript_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        return loss

    def compute_ad_loss(self, speech_encoder_out, text_encoder_out, reduce=True):
        if self.memory_num>0:
            input1 = speech_encoder_out["encoder_out"][0]
            input2 = text_encoder_out["encoder_out"][0]
            assert input1.shape == input2.shape
            input1 = input1.transpose(0, 1)
            input2 = input2.transpose(0, 1)  # [batch, seqlen, dim]
            if self.type == "contrastive":
                batch_size, seqlen, _ = input1.shape
                logits = torch.cosine_similarity(
                    input1.float().unsqueeze(2),
                    input2.float().unsqueeze(1),
                    dim=-1
                ).type_as(input1)
                logits /= self.contrastive_temp
                target = torch.arange(seqlen)[None].repeat(batch_size, 1) \
                    .to(logits.device)
                loss = F.cross_entropy(logits, target,
                                       reduction='sum' if reduce else "none")
            elif self.type == "mse":
                loss = 0.5 * (input1 - input2) ** 2
                loss = loss.sum(-1)
                if reduce:
                    loss = loss.sum()
            else:
                raise NotImplementedError
            return loss
        else:
            raise NotImplementedError

    def compute_mlm_loss(self, model, encoder_out, target):
        logits = model.encoder.compute_output_logits(encoder_out) # T x B x V
        logits = logits.transpose(0,1)
        V = logits.size(-1)
        loss = F.cross_entropy(
            logits.contiguous().view(-1, V), target.view(-1), reduction="sum"
        )
        return loss


    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(
            sum(log.get("loss", 0) for log in logging_outputs)
        )
        trans_loss_sum = utils.item(
            sum(log.get("trans_loss", 0) for log in logging_outputs)
        )
        nll_loss_sum = utils.item(
            sum(log.get("nll_loss", 0) for log in logging_outputs)
        )
        ctc_loss_sum = utils.item(
            sum(log.get("ctc_loss", 0) for log in logging_outputs)
        )
        mlm_loss_sum = utils.item(
            sum(log.get("mlm_loss", 0) for log in logging_outputs)
        )
        ad_loss_sum = utils.item(
            sum(log.get("ad_loss", 0) for log in logging_outputs)
        )
        text_ntokens = utils.item(sum(log.get("text_ntokens", 0) for log in logging_outputs))
        spch_ntokens = utils.item(sum(log.get("spch_ntokens", 0) for log in logging_outputs))
        mlm_ntokens = utils.item(sum(log.get("mlm_ntokens", 0) for log in logging_outputs))
        lengths = utils.item(sum(log.get("lengths", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if text_ntokens > 0:
            metrics.log_scalar(
                "trans_loss", trans_loss_sum / text_ntokens / math.log(2), text_ntokens, round=3
            )
            metrics.log_scalar(
                "nll_loss", nll_loss_sum / text_ntokens / math.log(2), text_ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        if spch_ntokens > 0:
            metrics.log_scalar(
                "ctc_loss", ctc_loss_sum / spch_ntokens / math.log(2), spch_ntokens, round=3
            )
        if mlm_ntokens > 0:
            metrics.log_scalar(
                "mlm_loss", mlm_loss_sum / mlm_ntokens / math.log(2), mlm_ntokens, round=3
            )
        if lengths > 0:
            metrics.log_scalar(
                "ad_loss", ad_loss_sum / lengths / math.log(2), lengths, round=3
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


