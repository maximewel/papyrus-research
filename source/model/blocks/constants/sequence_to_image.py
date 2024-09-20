import math
import numpy as np
import cv2

from source.model.blocks.constants.tokens import Tokens

class ImageHelper():

    DRAW_COLOR_WHITE = 0
    DRAW_COLOR_BLACK = 255
    DRAW_COLOR_SIZE = 1


    @classmethod
    def is_eos_token(cls, x, y):
        return x < 0 and y < 0
        return x == Tokens.COORDINATE_SEQUENCE_EOS and y == Tokens.COORDINATE_SEQUENCE_EOS

    @classmethod
    def create_image(cls, signal: list[int, int, bool]):
        """Create the image associated with the given signal."""
        max_h =  int(math.ceil(max(signal[:, 0])))
        max_w = int(math.ceil(max(signal[:, 1])))

        canvas = np.ascontiguousarray(np.full((max_w + 1, max_h + 1), cls.DRAW_COLOR_BLACK), dtype=np.uint8)

        #Draw lines from point (t-1) to current point (t) IFF the pen was not up. start with penup
        #as we start from point 0.
        draw_current_stroke = False
        for x, y, eos in signal:

            if draw_current_stroke and not cls.is_eos_token(x, y):
                cv2.line(canvas, (last_x, last_y), (x, y), cls.DRAW_COLOR_WHITE, cls.DRAW_COLOR_SIZE) 
            last_x, last_y, draw_current_stroke = x, y, not eos
        
        return canvas