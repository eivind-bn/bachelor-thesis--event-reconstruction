import numpy as np

from core.constants.colors import BLACK, WHITE, BLUE
from core.transformer.module import Transformer


class FramesByTimestamps(Transformer):

    def __init__(self,
                 void_color=BLACK,
                 pos_color=WHITE,
                 neg_color=BLUE,
                 fps=24.0):

        super().__init__()

        self.frame_buffer = None

        self.frame_period_us = (1 / fps) * 1e6
        self.frame_cntr = 1
        self.t_pointer = self.frame_period_us
        self.void_color = void_color
        self.pos_color = pos_color
        self.neg_color = neg_color

    def late_init(self, height, width, **kwargs):
        self.frame_buffer = np.zeros((height, width, 3), dtype=np.ubyte)

    def process_data(self, events, **kwargs):
        frame = np.argwhere(events['t'] < self.t_pointer)
        yi, xi, pi = events['y'][frame], events['x'][frame], events['p'][frame]

        zeroes = np.argwhere(pi == 0)
        ones = np.argwhere(pi == 1)

        self.frame_buffer[yi[zeroes], xi[zeroes]] = self.neg_color
        self.frame_buffer[yi[ones], xi[ones]] = self.pos_color

        if frame.size < events.size:
            self.callback(self.frame_buffer, **kwargs)
            self.frame_buffer[:, :] = self.void_color

            self.t_pointer = self.frame_cntr * self.frame_period_us
            self.frame_cntr += 1
            self.process_data(events[frame.size:])
