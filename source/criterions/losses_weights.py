class LossesWeights:
    """Simple class that allows the weights to be appleid to the losses"""
    eos_weight: int
    coord_weight: int
    skeleton_weight: int

    total_weights: int

    def __init__(self, eos_weight: int, coord_weight: int, skeleton_weight: int):
        self.eos_weight = eos_weight        
        self.coord_weight = coord_weight        
        self.skeleton_weight = skeleton_weight

        self.total_weights = eos_weight + coord_weight + skeleton_weight