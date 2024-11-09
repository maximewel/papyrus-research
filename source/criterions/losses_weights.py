class LossesWeights:
    """Simple class that allows the weights to be appleid to the losses"""
    coord_weight: int
    skeleton_weight: int

    total_weights: int

    def __init__(self, coord_weight: int, skeleton_weight: int):
        self.coord_weight = coord_weight        
        self.skeleton_weight = skeleton_weight

        self.total_weights = coord_weight + skeleton_weight