from pathlib import Path, PosixPath
from typing import Dict, Union

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from ASR.modules.pipeline.constants import PAD_IDX, TEST_PROCESSOR, TRAIN_PROCESSOR


class LibriDataset(Dataset):
    def __init__(
        self,
        root: PosixPath = Path('.'),
        split: str = 'dev-clean',
    ):
        assert split in [
            'dev-clean',
            'dev-other',
            'test-clean',
            'test-other',
            'train-clean-100',
        ], 'Split error!'

        self.data_iterator = torchaudio.datasets.LIBRISPEECH(root=root, url=split)

        if split.split('-')[0] == 'train':
            self.processor = TRAIN_PROCESSOR
        else:
            self.processor = TEST_PROCESSOR

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Union[torch.FloatTensor, torch.LongTensor]]:
        sample = self.data_iterator[idx]
        sample = self.processor(sample[0][0], sample[2].lower())

        return sample

    def __len__(self):
        return len(self.data_iterator)

    @staticmethod
    def normalize_batch(
        sample: torch.FloatTensor,
        input_lengths: torch.LongTensor,
    ) -> torch.FloatTensor:
        input_features = sample['input_features']

        for s_idx, s_len in enumerate(input_lengths):
            valid_features = input_features[s_idx, :, :s_len]
            sample_mean = torch.mean(valid_features)
            sample_std = torch.sqrt(torch.mean(valid_features**2) - sample_mean**2)

            input_features[s_idx, :, :s_len] = (
                valid_features - sample_mean
            ) / sample_std

        return input_features

    @staticmethod
    def batch_collate(
        batch: Dict[str, Union[torch.FloatTensor, torch.LongTensor]],
        text_pad_token: int,
    ) -> Dict[str, Union[torch.FloatTensor, torch.LongTensor]]:
        # Collate audio samples.
        sample_tokens_lengths = torch.tensor(
            [x['input_features'].size(1) for x in batch]
        )
        max_len_per_samples = torch.max(sample_tokens_lengths)

        # Extend to even max_len.
        additive = max_len_per_samples % 2
        max_len_per_samples += additive
        samples_lengths_to_pad = max_len_per_samples - sample_tokens_lengths - additive

        input_features = torch.stack(
            [
                F.pad(x['input_features'], pad=(0, val_to_pad))
                for x, val_to_pad in zip(batch, samples_lengths_to_pad)
            ]
        )

        # Collate label samples.
        label_tokens_lengths = torch.tensor([x['labels'].size(0) for x in batch])
        max_len_per_labels = torch.max(label_tokens_lengths)

        # Extend to even max_len.
        additive = max_len_per_labels % 2
        max_len_per_labels += additive
        labels_lengths_to_pad = max_len_per_labels - label_tokens_lengths - additive

        labels = torch.vstack(
            [
                F.pad(x['labels'], pad=(0, val_to_pad), value=text_pad_token)
                for x, val_to_pad in zip(batch, labels_lengths_to_pad)
            ]
        ).type(torch.int64)

        collated = {
            'input_features': input_features,
            'targets': labels,
            'attention_mask': (labels != PAD_IDX).type(torch.int64),
        }

        collated['input_features'] = LibriDataset.normalize_batch(
            collated, sample_tokens_lengths
        )

        return collated
