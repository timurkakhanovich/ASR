from typing import List, Tuple, Dict, Union
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ASR.modules.pipeline.constants import BPEMB_EN, DEVICE, T
from ASR.modules.model.model import QuartzLM


def concat_raw_arrays(
    old_seq: torch.LongTensor,
    new_seq: torch.LongTensor,
) -> torch.LongTensor:
    '''
    Parallel arrays concatenation:
    torch.tensor([1, 2])
    torch.tensor([5, 4, 3, 10])
    
    Result:
    tensor([[ 1,  2,  5],
        [ 1,  2,  4],
        [ 1,  2,  3],
        [ 1,  2, 10]])
    '''
    
    return torch.stack([
        torch.cat((old_seq, j)) for j in new_seq.unsqueeze(-1)
    ])


def concat_seqs(
    old_seq: torch.LongTensor,
    new_seq: torch.LongTensor,
) -> torch.LongTensor:
    '''
    Parallel sequences concatenation.
    '''
    
    return torch.stack([
        concat_raw_arrays(o, n) for o, n in zip(old_seq, new_seq)
    ])


def get_topk_in_matrix(
    matrix: torch.FloatTensor,
    k: int,
) -> Tuple[torch.LongTensor]:
    '''
    Search top k values in matrix
    
    Returns: indices of top k values in matrix.
    '''
    
    _, topk_ids = torch.topk(matrix.flatten(), k, dim=-1)
    
    return torch.div(topk_ids, k, rounding_mode='floor'), topk_ids % k


@torch.inference_mode()
def greedy_decoding(
    model: QuartzLM,
    sample: torch.FloatTensor,
    max_len: int,
) -> torch.LongTensor:
    training = model.training
    model.eval()
    
    batch_size = sample.size(0)
    predictions = [torch.full([batch_size], BPEMB_EN.BOS, 
                          dtype=torch.int64, device=DEVICE)]
    
    # Get initial state
    hx, cx = model.encoder(sample)
    for _ in range(max_len):
        logits, (hx, cx) = model.decoder.decode_step(hx, cx, predictions[-1])
        predictions.append(logits.argmax(dim=-1))
        
    if training:
        model.train()
    
    return torch.stack(predictions, dim=1)


@torch.inference_mode()
def start_beam_search(
    model: QuartzLM,
    h0: torch.FloatTensor,
    c0: torch.FloatTensor,
    beam_size: int,
) -> Dict[str, Union[torch.LongTensor, torch.FloatTensor, Tuple[torch.FloatTensor]]]:
    '''
    Encode sample (melspec), predict the first token of sentence 
    after BOS token and return the next hidden states

    Returns: 
    dict of:
    * the top k sequnces,
    * the top k of their probabilities,
    * their hidden and cell states of LSTM
    '''

    result = {
        'seq': torch.full([beam_size], BPEMB_EN.BOS, dtype=torch.int64, device=DEVICE),
        'log_probs': torch.zeros(beam_size, dtype=torch.float32, device=DEVICE)
    }
    init_seq = torch.tensor([BPEMB_EN.BOS], dtype=torch.int64, device=DEVICE)

    pred1, (h1, c1) = model.decoder.decode_step(h0, c0, init_seq)
    top1 = torch.topk(pred1, k=beam_size, dim=-1)

    result['seq'] = torch.vstack((result['seq'], top1.indices[0])).T
    result['log_probs'] -= top1.values[0]
    result['states'] = [(h1, c1)] * beam_size

    return result


@torch.inference_mode()
def beam_search_loop(
    model: QuartzLM,
    result: Dict[str, Union[torch.LongTensor, torch.FloatTensor, Tuple[torch.FloatTensor]]],
    beam_size: int,
    max_len: int,
) -> torch.LongTensor:
    # Max len excluding BOS and start_iteration
    for _ in range(max_len - 2):
        temp_result = defaultdict(list)

        for i in range(beam_size):
            curr_token = torch.tensor([result['seq'][i, -1]], dtype=torch.int64, device=DEVICE)

            pred, states = model.decoder.decode_step(*result['states'][i], curr_token)

            top = torch.topk(pred, k=beam_size, dim=-1)

            temp_result['log_probs'].append(result['log_probs'] - top.values.squeeze())
            temp_result['seq'].append(top.indices[0])
            temp_result['states'].append(states)  # states = (h, c)

        temp_result['log_probs'] = torch.stack(temp_result['log_probs'])
        temp_result['seq'] = concat_seqs(result['seq'], torch.stack(temp_result['seq']))
        top_ids = get_topk_in_matrix(
            temp_result['log_probs'], k=beam_size
        )
        result['log_probs'] = temp_result['log_probs'][top_ids]
        result['seq'] = temp_result['seq'][top_ids]
        result['states'] = [temp_result['states'][idx]
                            for idx in top_ids[0]]

    high_p_idx = torch.argmax(result['log_probs'])

    return result['seq'][high_p_idx]


@torch.inference_mode()
def beam_search_decoding(
    model: QuartzLM,
    sample: torch.FloatTensor,
    beam_size: int = 4,
    max_len: int = 100,
) -> torch.LongTensor:
    training = model.training
    model.eval()
    
    batch_size = sample.size(0)

    # Get initial state
    h0, c0 = model.encoder(sample)
    predictions = torch.zeros(batch_size, max_len, dtype=torch.int64)
    for batch_idx in range(batch_size):
        hx = h0[batch_idx].unsqueeze(0)
        cx = c0[batch_idx].unsqueeze(0)

        result = start_beam_search(model, hx, cx, beam_size)
        predictions[batch_idx] = beam_search_loop(model, result, beam_size, max_len)
    
    if training:
        model.train()
    
    return predictions


def translate_lines(
    input_lines: torch.LongTensor,
) -> List[str]:
    result_str = []
    for i in range(input_lines.size(0)):
        bpe_format_str = \
            ''.join([BPEMB_EN.emb.index_to_key[j] for j in input_lines[i][1:]])
        substring_before_eos = bpe_format_str.split(BPEMB_EN.EOS_str)[0]

        result_str.append(' '.join(substring_before_eos.split('▁')).strip())

    return result_str


@dataclass
class ValidateOut:
    loss: float
    metrics: Dict[str, float]


@torch.inference_mode()
def validate_model(
    model: QuartzLM,
    val_dataloader: DataLoader,
    criterion: nn.modules.loss,
    metrics: T,
) -> ValidateOut:
    training = model.training
    model.eval()
    
    running_loss = 0.0
    running_score = defaultdict(float)
    
    print('\n')
    for batch_idx, sample in enumerate(val_dataloader):
        if batch_idx % 10 == 0 or batch_idx == len(val_dataloader) - 1:
            print(f'==> Batch: {batch_idx}/{len(val_dataloader)}')

        sample = {k: v.to(DEVICE) for k, v in sample.items()}
        
        sample['log_probs'] = model(sample['input_features'], sample['targets'])
        
        metrics_info = metrics.val_metrics(sample['input_features'], sample['targets'])
        loss = criterion(sample['log_probs'], sample['targets'])
        del sample['input_features']
        
        running_loss += loss.item()
        running_score['greedy_wer'] += metrics_info.wer['greedy']
        running_score['bs_wer'] += metrics_info.wer['beam_search']

    running_loss /= len(val_dataloader)
    running_score = {
        k: v / len(val_dataloader) for k, v in running_score.items()
    }
    
    if training:
        model.train()
    
    return ValidateOut(running_loss, running_score)
