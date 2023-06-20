from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_metric

from ASR.modules.model.model import QuartzLM
from ASR.modules.pipeline.inference import (
    beam_search_decoding,
    greedy_decoding,
    translate_lines,
)


@dataclass
class MetricsOut:
    pred_str: Dict[str, List[float]]
    label_str: str
    wer: Dict[str, List[float]]


class Metrics:
    def __init__(
        self,
        model: QuartzLM,
        beam_size: int = 4,
        max_len: int = 100,
    ):
        super().__init__()
        self.model = model
        self.wer_metric = load_metric('wer')

        self.beam_size = beam_size
        self.max_len = max_len

    def train_metrics(
        self,
        pred: torch.FloatTensor,
        label_str: str,
    ) -> MetricsOut:
        pred_str = translate_lines(pred.argmax(dim=1))
        wer = {
            'greedy': self.wer_metric.compute(
                predictions=pred_str, references=label_str
            )
        }
        return MetricsOut(pred_str, label_str, wer)

    def val_metrics(
        self,
        sample: torch.FloatTensor,
        label_str: str,
    ) -> MetricsOut:
        greedy_pred = greedy_decoding(self.model, sample, self.max_len)
        bs_pred = beam_search_decoding(self.model, sample, self.beam_size, self.max_len)

        pred_str = {
            'greedy': translate_lines(greedy_pred),
            'beam_search': translate_lines(bs_pred),
        }
        wer = {
            k: self.wer_metric.compute(predictions=pred, references=label_str)
            for k, pred in pred_str.items()
        }

        return MetricsOut(pred_str, label_str, wer)
