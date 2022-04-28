from pathlib import Path
from functools import partial

import torch
import torch.nn as nn

import wandb

from ASR.modules.data.dataset import LibriDataset
from ASR.modules.data.split_data import set_split
from ASR.modules.model.model import QuartzLM
from ASR.modules.pipeline.utils import set_seed
from ASR.modules.pipeline.constants import BPEMB_EN, PAD_IDX, N_MELS, HID_SIZE, NUM_LAYERS
from ASR.modules.pipeline.model_config import get_arguments, load_from_checkpoint
from ASR.modules.pipeline.metrics import Metrics
from ASR.modules.pipeline.train import train


def main_command_line_args() -> None:
    set_seed(42)
    CHECKPOINT_PATH = Path('ASR_checkpoints/')
    
    model = QuartzLM(N_MELS, HID_SIZE, NUM_LAYERS)
    configs = get_arguments(model)
    
    device = configs['device']
    model = configs['model']
    
    optimizer = configs['optimizer']
    scheduler = configs['scheduler']
    num_epochs = configs['num_epochs']
    start_epoch = configs['start_epoch']
    history = configs['history']
    lr = configs['lr']
    batch_size = configs['batch_size']
    
    collator = partial(LibriDataset.batch_collate, text_pad_token=PAD_IDX)
    data = set_split(
        batch_size=batch_size, train_shuffle=True, collator=collator
    )
    
    metrics = Metrics(model, beam_size=4, max_len=100)
    criterion = nn.NLLLoss()
    
    config = {
        'learning_rate': lr,
        'batch_size': batch_size,
        'embedding_dim': BPEMB_EN.dim,
        'hid_size': model.hid_size
    }
    
    with wandb.init(project='ASR', entity='timkakhanovich', config=config) as run:
        wandb.watch(model, log='all', log_freq=150)
        
        model_artifact = wandb.Artifact(
            'QuartzLM', type='model'
        )
        model_filename = 'model.pt'
        torch.save(model.state_dict(), model_filename)
        model_artifact.add_file(model_filename)

        wandb.save(model_filename)
    
        train(
            model, data.dataloaders, criterion, optimizer, metrics, scheduler,
            num_epochs, start_epoch, prev_metrics=history, device=device,
            folder_for_checkpoints=CHECKPOINT_PATH, run=run
        )

if __name__ == '__main__':
    main_command_line_args()
