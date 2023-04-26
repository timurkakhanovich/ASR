from typing import TypeVar

from bpemb import BPEmb
import torch

from ASR.modules.data.data_processing import ASRProcessor


# Tokenizer.  
BPEMB_EN = BPEmb(lang='en', dim=300, vs=1000)
PAD_IDX = BPEMB_EN.EOS

# Data processor.  
TRAIN_PROCESSOR = ASRProcessor(BPEMB_EN, split='train')
TEST_PROCESSOR = ASRProcessor(BPEMB_EN, split='test')

# Training.  
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_MELS = 64
HID_SIZE = 300
NUM_LAYERS = 2

# Var type.  
T = TypeVar('T')
