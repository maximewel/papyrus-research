"""Module used to regroup some functions shared to operate across tensors"""

from torch import Tensor

from source.model.blocks.constants.tokens import Tokens

class TensorUtils():
    
    @classmethod
    def padding_indexes_mask(cls, tensor: Tensor) -> Tensor:
        """Return the padding indexes mask """
        return ~(tensor == Tokens.EOS_TENSOR.value.to(tensor.device)).all(dim=-1)
    
    @classmethod
    def true_seq_lengths_of_tensor(cls, tensor: Tensor) -> tuple:
        """Compute the true seq length of the given tensor, each row without padding"""
        padding_mask = TensorUtils.padding_indexes_mask(tensor)
        return padding_mask.sum().item()

    @classmethod
    def true_seq_lengths_of_tensors(cls,tensor: Tensor) -> tuple:
        """Compute the true seq length of the given tensor, each row without padding"""
        shapes = []
        for i in range(tensor.shape[0]):
            tensor_row = tensor[i]
            cleaned_row = tensor_row[TensorUtils.padding_indexes_mask(tensor_row)]
            shapes.append(cleaned_row.shape[0])
        return shapes

    @classmethod
    def min_max_sequence_len_in_tensor(cls, tensor: Tensor) -> tuple[int, int]:
        """Determine the max sequence length (non-padded) in the given tensor"""
        shapes = TensorUtils.true_seq_lengths_of_tensor(tensor)
        return min(shapes), max(shapes)