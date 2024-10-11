import numpy as np 
import torch

src = torch.tensor([[1, 2, 3, 4], [0, 0, 0, 0]])

print(f"Source: {src.shape} \n{src}")
print(f"As mask: {(src != 0).shape} \n{src != 0}")
print(f"First unsqueeze on first dimension: {(src != 0).unsqueeze(1).shape} \n{(src != 0).unsqueeze(1)}")
print(f"Second unsqueeze on second dim: {(src != 0).unsqueeze(1).unsqueeze(2).shape} \n{(src != 0).unsqueeze(1).unsqueeze(2)}")
print(f"Second unsqueeze on second dim: {(src != 0).unsqueeze(1).unsqueeze(1).shape} \n{(src != 0).unsqueeze(1).unsqueeze(1)}")

src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

p1 = [2, 2]
p2 = [3, 3]

p3 = [4, 4]
p4 = [5, 5]

p5 = [-1, -1]
p6 = [-1, -1]

# hidden_d = 2
# seq:len = 2
# batch_size = 3

batched_input = [[p1, p2], [p3, p4], [p5, p6]]

print(torch.tensor(batched_input).shape)

target_seq = torch.tensor([[p1, p2, p3, p4, p5, p6]])

print(target_seq.shape)


padding_mask = (target_seq == (-1)).all(dim=-1)
print(f"Padding mask: {padding_mask.shape}\n{padding_mask}")

seq_length = target_seq.size(1)
trig_matrix = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
nope_mask = trig_matrix.bool()

print(f"Nop_mask: {nope_mask.shape} \n{nope_mask}")


# from source.model.blocks.constants.tokens import Tokens

# batched_input_with_token = torch.tensor([[p1, p2], [p3, p4], [p5, Tokens.COORDINATE_SEQUENCE_PADDING_TOKEN.value]])
# print(batched_input_with_token)
# padding_mask_with_token = (batched_input_with_token == Tokens.COORDINATE_SEQUENCE_PADDING_TOKEN.value).all(dim=-1)
# print(f"Padding mask: {padding_mask_with_token.shape}\n{padding_mask_with_token}")

EOS_TENSOR = torch.Tensor([0, 0])

c = torch.Tensor([
    [1, 2],
    [3, 4],
    [5, 6],
    EOS_TENSOR,
    EOS_TENSOR,
    EOS_TENSOR,
    EOS_TENSOR,
    EOS_TENSOR
])

print(c.shape)
print((c ==EOS_TENSOR).all(dim=-1).sum())
print((c ==EOS_TENSOR).all(dim=-1))
print(c)

m = ~(c == EOS_TENSOR).all(dim=-1)
print(m.sum().item())