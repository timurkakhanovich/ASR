from dataclasses import dataclass
from typing import Dict, Union, Optional

from torch.utils.data import DataLoader

from ASR.modules.data.dataset import LibriDataset
from ASR.modules.pipeline.constants import BPEMB_EN, T


@dataclass
class Data:
    datasets: Dict[str, LibriDataset]
    dataloaders: Dict[str, DataLoader]


def set_split(
    batch_size: int = 32,
    train_shuffle: bool = False,
    collator: Optional[T] = None,
) -> Data:    
    datasets = {
        'train': LibriDataset(split='train-clean-100'), 
        'val': LibriDataset(split='dev-clean'), 
        'test': LibriDataset(split='test-clean')
    }
    
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=train_shuffle, 
                                collate_fn=collator, num_workers=4), 
        'val': DataLoader(datasets['val'], batch_size=32, shuffle=False, 
                            collate_fn=collator, num_workers=4), 
        'test': DataLoader(datasets['test'], batch_size=32, shuffle=False, 
                            collate_fn=collator, num_workers=4)
    }
    
    return Data(datasets, dataloaders)
