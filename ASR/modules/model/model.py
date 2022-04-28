from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ASR.modules.pipeline.constants import BPEMB_EN, PAD_IDX


class SingleBBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        activation: bool = True,
    ):
        super().__init__()
        
        # Padding 'same'.  
        padding = (kernel_size // 2) * dilation
        
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size, stride, 
            padding=padding, dilation=dilation, groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
        self.activation = activation
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        TCS_out = self.pointwise(self.depthwise(x))
        
        bn_out = self.batch_norm(TCS_out)
        
        return F.relu(bn_out) if self.activation else bn_out


class RepeatedBBlocks(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        R: int = 5,
    ):
        super().__init__()
        
        # The first block to match in_channels and out_channels.  
        self.B = [
            SingleBBlock(
                in_channels, out_channels, kernel_size, 
                stride, dilation, activation=True
            )
        ]
        
        # BBlocks between the first and the last blocks.  
        self.B.extend([
            SingleBBlock(
                out_channels, out_channels, kernel_size, 
                stride, dilation, activation=True
            )
            for _ in range(R - 2)
        ])
        
        # The last block to prevent nonlinearity.  
        self.B.append(
            SingleBBlock(
                out_channels, out_channels, kernel_size, 
                stride, dilation, activation=False
            )
        )
        self.B = nn.Sequential(*self.B)
        
        # Skip connection.  
        self.skip_connection = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1), 
            nn.BatchNorm1d(num_features=out_channels)
        )
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        RBlocks_out = self.B(x)
        skip_out = self.skip_connection(x)
        
        return F.relu(RBlocks_out + skip_out)


class QuartzNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        
        self.C1 = SingleBBlock(
            in_channels=in_features, out_channels=256, kernel_size=33, 
            stride=2, dilation=1, activation=True
        )
        
        self.B = nn.Sequential(
            OrderedDict([
                ('B1', RepeatedBBlocks(
                    in_channels=256, out_channels=256, kernel_size=33, 
                    stride=1, dilation=1, R=5
                )), 
                ('B2', RepeatedBBlocks(
                    in_channels=256, out_channels=256, kernel_size=39, 
                    stride=1, dilation=1, R=5
                )), 
                ('B3', RepeatedBBlocks(in_channels=256, out_channels=512, kernel_size=51, 
                                       stride=1, dilation=1, R=5
                )), 
                ('B4', RepeatedBBlocks(in_channels=512, out_channels=512, kernel_size=63, 
                                       stride=1, dilation=1, R=5
                )), 
                ('B5', RepeatedBBlocks(in_channels=512, out_channels=512, kernel_size=75, 
                                       stride=1, dilation=1, R=5
                ))
            ])
        )
        self.C2 = SingleBBlock(
            in_channels=512, out_channels=512, kernel_size=87, 
            stride=1, dilation=2, activation=True
        )
        
        self.C3 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, dilation=1), 
            nn.BatchNorm1d(num_features=1024), 
            nn.ReLU()
        )
        self.C4 = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=out_features, kernel_size=1, stride=1, dilation=1), 
            nn.BatchNorm1d(num_features=out_features), 
            nn.ReLU()
        )
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        first_conv_out = self.C1(x)

        b_out = self.B(first_conv_out)
        c2_out = self.C2(b_out)
        c3_out = self.C3(c2_out)
        c4_out = self.C4(c3_out)
        
        return c4_out
    

class Encoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        hid_size: int,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.acoustic_model = QuartzNet(in_features, hid_size)
        self.lstm_enc = nn.LSTM(
            input_size=hid_size, hidden_size=hid_size, 
            num_layers=num_layers, batch_first=True, dropout=0.2
        )
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        acoustic_out = self.acoustic_model(x).permute(0, 2, 1)
        
        _, (h, c) = self.lstm_enc(acoustic_out)  # x: [B, SEQ, H]
        
        return h[-1], c[-1]
    

class Decoder(nn.Module):
    def __init__(
        self,
        hid_size: int,
    ):
        super().__init__()
        
        self.emb_out = nn.Embedding(num_embeddings=BPEMB_EN.vs, embedding_dim=BPEMB_EN.dim, 
                                    padding_idx=PAD_IDX)
        self.emb_out.weight.data.copy_(torch.from_numpy(BPEMB_EN.vectors))
        
        self.lstm_dec = nn.LSTMCell(input_size=BPEMB_EN.dim, hidden_size=hid_size)
        self.logits = nn.Linear(in_features=hid_size, out_features=BPEMB_EN.vs)
    
    def decode_step(
        self,
        h_prev: torch.FloatTensor,
        c_prev: torch.FloatTensor,
        cur_token: torch.LongTensor,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor]]:
        emb_target = self.emb_out(cur_token)
        
        hx, cx = self.lstm_dec(emb_target, (h_prev, c_prev))
        logits = F.log_softmax(self.logits(hx), dim=-1)
        
        return logits, (hx, cx)
        
    def forward(
        self,
        hx: torch.FloatTensor,
        cx: torch.FloatTensor,
        target: torch.LongTensor,
    ) -> torch.FloatTensor:
        seq_first_target = target.T  # (S, B)
        
        predictions = []
        for curr_token in seq_first_target:
            logits, (hx, cx) = self.decode_step(hx, cx, curr_token)
            predictions.append(logits)
        
        return torch.stack(predictions)
    
    
class QuartzLM(nn.Module):
    def __init__(
        self,
        in_features: int,
        hid_size: int,
        num_layers: int,
    ):
        super().__init__()
        
        self.encoder = Encoder(in_features, hid_size, num_layers)
        self.decoder = Decoder(hid_size)
        
        self.hid_size = hid_size
        
    def forward(
        self,
        inp: torch.FloatTensor,
        target: torch.LongTensor,
    ) -> torch.FloatTensor:
        hx, cx = self.encoder(inp)
        pred_sequence = self.decoder(hx, cx, target)
        
        return pred_sequence.permute(1, 2, 0)
