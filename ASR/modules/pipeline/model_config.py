from pathlib import PosixPath
from typing import Dict, Any
import argparse
import textwrap

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from ASR.modules.model.model import QuartzLM
from ASR.modules.pipeline.constants import DEVICE


def load_from_checkpoint(
    model: QuartzLM,
    checkpoint_path: PosixPath,
    check_epoch: int,
    batch_size: int,
    num_epochs: int,
) -> Dict[str, Any]:
    full_check_path = checkpoint_path / f'checkpoint_epoch_{check_epoch}.pt'
    checkpoint = torch.load(full_check_path, map_location=DEVICE)

    epoch = checkpoint['epoch']
    model = model.to(DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'])

    lr = checkpoint['lr']
    history = checkpoint['whole_history']

    optimizer = AdamW(model.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if checkpoint['scheduler_state_dict']:
        scheduler = StepLR(optimizer)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler if checkpoint['scheduler_state_dict'] else None,
        'history': history,
        'batch_size': batch_size,
        'start_epoch': epoch,
        'lr': lr,
        'num_epochs': num_epochs,
        'device': DEVICE,
    }
    

def get_arguments(model: QuartzLM) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog='PROG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''
        Gets arguments from command line to initialize model.
        STRICTLY REQUIRED: 
        * for chosen checkpoint: CHECKPOINT, NUM_EPOCHS;
        * if checkpoint is not chosen: LR, BATCH_SIZE, NUM_EPOCHS
        ''')
    )
    parser.add_argument('--checkpoint', type=int, help='Loading from checkpoint')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, help='Num epochs for training')

    args = parser.parse_args()

    if args.checkpoint:
        return load_from_checkpoint(check_epoch=args.checkpoint, 
                                    batch_size=args.batch_size,
                                    num_epochs=args.num_epochs, 
                                    device=DEVICE)
    else:
        model = model.to(DEVICE)

        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = None

        return {
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'history': {},
            'batch_size': args.batch_size,
            'start_epoch': -1,
            'lr': args.lr,
            'num_epochs': args.num_epochs,
            'device': DEVICE,
        }
