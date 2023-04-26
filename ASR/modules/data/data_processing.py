from typing import Dict, Union, Any

import torch
import torch.nn as nn
import torchaudio


class Unsqueeze(nn.Module):
    def __init__(self, dim: int = 0):
        super().__init__()
        
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.FloatTensor:
        return x.unsqueeze(self.dim)


class ASRProcessor:
    def __init__(
        self,
        tokenizer: Any,
        sampling_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 64,
        split: str = 'val',
    ):
        self.mel_spec_processor = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            n_mels=n_mels
        )
        
        if split == 'train':
            self.augmentation = nn.Sequential(
                Unsqueeze(), 
                torchaudio.transforms.TimeMasking(time_mask_param=5), 
                torchaudio.transforms.FrequencyMasking(freq_mask_param=5), 
            )
        else:
            self.augmentation = None
        
        self.tokenizer = tokenizer
        
    def text_processor(self, text: str) -> torch.LongTensor:
        encoded_text = [self.tokenizer.BOS] + \
                       self.tokenizer.encode_ids(text) + \
                       [self.tokenizer.EOS]
        
        return torch.tensor(encoded_text, dtype=torch.int64)
    
    def __call__(
        self,
        input_values: torch.FloatTensor,
        labels: str,
    ) -> Dict[str, Union[torch.FloatTensor, torch.LongTensor]]:
        # Got such boundaries for 99.8% of non-augmented train data: [-10.7776, 6.4294].  
        log_mel_spec_image = torch.log(self.mel_spec_processor(input_values) + 1e-6).clamp_(-10, 6)
        input_preprocessed = self.augmentation(log_mel_spec_image).squeeze(0) \
                                if self.augmentation else log_mel_spec_image
        
        text_preprocessed = self.text_processor(labels)
        
        return {
            'input_features': input_preprocessed, 
            'labels': text_preprocessed
        }
