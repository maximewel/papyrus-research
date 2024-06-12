"""Contains some comon help for encoder-decoder activation functions"""
from enum import Enum, auto
import torch.nn as nn

class FFActivationFunction(Enum):
    RELU = auto()
    GELU = auto()
    SILU = auto()
    LEAKYRELU = auto()

class ActivationHelper():
    ### Const ###
    LRELU_NEGATIVE_SLOPE = 0.1
    
    @classmethod
    def activation_from_enum(cls, FF_enum_value: FFActivationFunction) -> nn.Module:
        """Infer the activation function from the enum value"""
        match FF_enum_value:
            case FFActivationFunction.RELU:
                return nn.ReLU()
            case FFActivationFunction.GELU:
                return nn.GELU()
            case FFActivationFunction.SILU:
                return nn.SiLU()
            case FFActivationFunction.LEAKYRELU:
                return nn.LeakyReLU(negative_slope = cls.LRELU_NEGATIVE_SLOPE)
            case _:
                raise Exception(f"Impossible to parse activation function {FF_enum_value} into options {[f.name for f in FFActivationFunction]}")
