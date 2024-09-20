from enum import Enum
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Tokens(Enum):
    #A target sequences is composed of (x,y) coordinates. For not-yet-generated sequences, some have to be padded. 
    COORDINATE_SEQUENCE_PADDING_TOKEN = -0.5
    PADDING_TENSOR = torch.tensor([COORDINATE_SEQUENCE_PADDING_TOKEN, COORDINATE_SEQUENCE_PADDING_TOKEN], device=device)
    
    COORDINATE_SEQUENCE_EOS = -1
    EOS_TENSOR = torch.tensor([COORDINATE_SEQUENCE_EOS, COORDINATE_SEQUENCE_EOS], device=device)