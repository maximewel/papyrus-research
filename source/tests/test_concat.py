import torch

batch_size = 8
seq_len = 5
seq_dim = 2

sequences = torch.ones((batch_size, seq_len, seq_dim))

original_index = torch.arange(batch_size)

pred_seq_a = [torch.Tensor([1, 2]), torch.Tensor([3, 4]), torch.Tensor([5, 6]), torch.Tensor([7, 8]), torch.Tensor([9, 10])]

output_tensors = torch.zeros_like(sequences)

sequence_as_tensor = torch.cat(pred_seq_a, dim=0).view(-1, 2)
output_tensors[0, :len(sequence_as_tensor)] = sequence_as_tensor[:]

print(sequence_as_tensor)
print()
print(output_tensors)