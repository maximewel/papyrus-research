import torch
import torch.nn as nn 

class EuclideanDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """Euclidian distance is simply Sum(sqrt((Xi - Xj)**2))
        applied to each coordinate pair (x,y) between target and pred"""
        return torch.mean(torch.sqrt(torch.sum((pred - target) ** 2, dim=-1)))
    
if __name__ == "__main__":
    #Test
    criterion = EuclideanDistanceLoss()

    pred = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    target = torch.tensor([[1.0, 1.0], [4.0, 4.0]])
    loss = criterion(pred, target)
    print(f"Squared Euclidean Loss: {loss.item()}")

    pred = torch.tensor([[0, 0]])
    target = torch.tensor([[1.0, 1.0]])
    loss = criterion(pred, target)
    print(f"Squared Euclidean Loss: {loss.item()}")

    pred = torch.tensor([[0, 0], [0, 0]])
    target = torch.tensor([[1.0, 0], [0, 1.0]])
    loss = criterion(pred, target)
    print(f"Squared Euclidean Loss: {loss.item()}")