import math
import numpy as np
import cv2

class ImageHelper():

    DRAW_COLOR_WHITE = 0
    DRAW_COLOR_BLACK = 255
    DRAW_COLOR_SIZE = 1

    @classmethod
    def create_image(cls, signal: list[int, int, bool], canvas_size: tuple = None):
        """Create the image associated with the given signal."""
        if canvas_size is None:
            max_h =  int(math.ceil(max(signal[:, 0])))
            max_w = int(math.ceil(max(signal[:, 1])))
        else:
            max_w, max_h = canvas_size

        canvas = np.ascontiguousarray(np.full((max_w + 1, max_h + 1), cls.DRAW_COLOR_BLACK), dtype=np.uint8)

        #Draw lines from point (t-1) to current point (t) IFF the pen was not up. start with penup
        #as we start from point 0.
        draw_current_stroke = False
        for x, y, eos in signal:
            if draw_current_stroke:
                cv2.line(canvas, (last_x, last_y), (x, y), cls.DRAW_COLOR_WHITE, cls.DRAW_COLOR_SIZE) 
            last_x, last_y, draw_current_stroke = x, y, not eos
        
        return canvas