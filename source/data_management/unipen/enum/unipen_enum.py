from enum import Enum

class UnipenKeywords(Enum):
    # Config-related
    X_DIM = "X_DIM"
    Y_DIM = "Y_DIM"
    X_POINTS_PER_INCH = "X_POINTS_PER_INCH"
    Y_POINTS_PER_INCH = "Y_POINTS_PER_INCH"
    X_POINTS_PER_MM = "X_POINTS_PER_MM"
    Y_POINTS_PER_MM = "Y_POINTS_PER_MM"

    POINTS_PER_SECOND = "POINTS_PER_SECOND"
    COORD = "COORD"

    # Stroke-related
    PEN_UP = "PEN_UP"
    PEN_DOWN = "PEN_DOWN"
    START_BOX = "START_BOX"