from collections import defaultdict
from copy import deepcopy
from pathlib import Path, PosixPath
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from IPython.display import clear_output
from torch.utils.data import DataLoader
from tqdm import tqdm

from ASR.modules.model.model import QuartzLM
from ASR.modules.pipeline.constants import T
from ASR.modules.pipeline.inference import validate_model
from ASR.modules.pipeline.metrics import Metrics


def train(
    model: QuartzLM,
    dataloaders: DataLoader,
    criterion: nn.modules.loss,
    optimizer: optim,
    metrics: Metrics,
    scheduler: Optional[T] = None,
    num_epochs: int = 5,
    start_epoch: int = -1,
    prev_metrics: Optional[Dict[str, float]] = None,
    device: torch.device = torch.device('cpu'),
    folder_for_checkpoints: PosixPath = Path('.'),
    run: Optional[wandb.sdk.wandb_run.Run] = None,
) -> None:
    for key, vals in prev_metrics.items():
        for val in vals:
            wandb.log({key: val[1]}, step=val[0])

    if prev_metrics is not None:
        history = deepcopy(prev_metrics)
        curr_step = prev_metrics['train_loss'][-1][0] + 1
    else:
        history = defaultdict(list)
        curr_step = 1

    update_log_iteration = len(dataloaders['train']) // 2
    train_dataloader_len = len(dataloaders['train'])

    model.train()
    for epoch in range(start_epoch + 1, start_epoch + 1 + num_epochs):
        running_loss = 0.0
        running_score = 0.0

        clear_output(True)

        print("-" * 20)
        print(f"Epoch: {epoch}/{start_epoch + num_epochs}")
        print("-" * 20)
        print("Train: ")
        for batch_idx, sample in enumerate(tqdm(dataloaders['train'])):
            sample = {k: v.to(device) for k, v in sample.items()}

            sample['log_probs'] = model(sample['input_features'], sample['targets'])
            del sample['input_features']

            loss = criterion(sample['log_probs'], sample['targets'])
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            running_loss += loss.item()

            model_out = metrics.train_metrics(sample['log_probs'], sample['targets'])
            running_score += model_out.wer['greedy']

            if (
                batch_idx % (update_log_iteration + 1) == update_log_iteration
                or batch_idx == train_dataloader_len - 1
            ):
                val_result = validate_model(
                    model, dataloaders['val'], criterion, metrics
                )

                wandb.log(
                    {
                        'val_loss': val_result.loss,
                        'train_loss': running_loss / (batch_idx + 1),
                        'train_greedy_wer': running_score / (batch_idx + 1),
                    },
                    step=curr_step,
                )

                history['train_loss'].append(
                    (curr_step, running_loss / (batch_idx + 1))
                )
                history['val_loss'].append((curr_step, val_result.loss))
                history['train_greedy_wer'].append(
                    (curr_step, running_score / (batch_idx + 1))
                )

                for metric, score in val_result.metrics.items():
                    wandb.log({'val_' + metric: score}, step=curr_step)
                    history['val_' + metric].append((curr_step, score))

                curr_step += 1

        state = {
            'epoch': epoch,
            'batch_size_training': dataloaders['train'].batch_size,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'whole_history': history,
        }

        curr_checkpoint_path = (
            folder_for_checkpoints / f'checkpoint_epoch_{epoch%5 + 1}.pt'
        )
        torch.save(state, curr_checkpoint_path)

        model_art = wandb.Artifact('checkpoints', type='train_state')
        model_art.add_file(curr_checkpoint_path)
        run.log_artifact(model_art)
