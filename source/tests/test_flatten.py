import torch

a = [[1, 2], [3, 4], [5, 6]]
b = [[10, 20], [30, 40], [50, 60]]
c = [[100, 200], [300, 400], [500, 600]]


as_tensor = torch.tensor([a, b, c])

print(as_tensor.shape)

decoder_out_1 = as_tensor.view(3, -1)  
decoder_out_2 = torch.flatten(as_tensor, start_dim=1)  


print(decoder_out_1)

print(decoder_out_2)