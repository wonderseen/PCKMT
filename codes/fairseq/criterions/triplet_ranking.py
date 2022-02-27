# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion




@register_criterion("triplet_ranking")
class TripletRankingLoss(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=True,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

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
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        dis_ranking_loss, nce_lprobs, nce_loss, pivot, positive, negative = net_output
        sample_size = (
            pivot.size(0)
        )
        total_loss = dis_ranking_loss + nce_loss

        logging_output = {
            "loss": total_loss.data,
            "dis_ranking_loss": dis_ranking_loss.data,
            "nce_loss": nce_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            acc, n_correct, total = self.compute_accuracy(nce_lprobs)
            # logging_output["n_correct"] = utils.item(n_correct.data)
            # logging_output["total"] = utils.item(total.data)
            logging_output["acc"] = utils.item(acc.data)
            
        return total_loss, sample_size, logging_output

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

    def compute_accuracy(self, nce_lprobs):
        target = torch.arange(nce_lprobs.size(0)).to(nce_lprobs.device)
        predict = nce_lprobs.argmax(-1)
        n_correct = (target == predict).float()
        total = nce_lprobs.size(0)
        acc = n_correct.sum(-1) / nce_lprobs.size(0)
        return acc, n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get("loss", 0) for log in logging_outputs)
        loss_sum = sum(log.get("dis_ranking_loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nce_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        acc = sum(log.get("acc", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "dis_ranking_loss", loss_sum / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nce_loss", nll_loss_sum / sample_size / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nce_loss"].avg)
        )

        metrics.log_scalar(
            "acc", acc, sample_size, round=3
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
