import numpy as np

from core.link.module import Algorithm
from core.colors import BLUE, WHITE, BLACK
import time


class SequentialFrameGenerator(Algorithm):

    def __init__(self, width, height, **kwargs):
        super().__init__()
        self.width = width
        self.height = height
        self.screen_buffer = np.full((height, width, 3), BLACK, dtype=np.ubyte)
        self.frame_period_us = (1/kwargs.get('fps', 30.0))*1e6
        self.frame_cntr = 1
        self.t_pointer = self.frame_period_us
        self.t_mark = time.time_ns()*1e-3
        self.void_color = kwargs.get('void_color', BLACK)
        self.pos_color = kwargs.get('pos_color', WHITE)
        self.neg_color = kwargs.get('neg_color', BLUE)

    def frame_mark(self):
        delta_time_us = time.time_ns()*1e-3 - self.t_mark
        self.t_mark += delta_time_us
        sleep_time = self.frame_period_us - delta_time_us
        if sleep_time > 0:
            time.sleep(sleep_time*1e-6)

    def process_data(self, events: np.ndarray):
        frame = np.argwhere(events['t'] < self.t_pointer)
        yi, xi, pi = events['y'][frame], events['x'][frame], events['p'][frame]

        zeroes = np.argwhere(pi == 0)
        ones = np.argwhere(pi == 1)

        self.screen_buffer[yi[zeroes], xi[zeroes]] = self.neg_color
        self.screen_buffer[yi[ones], xi[ones]] = self.pos_color

        if frame.size < events.size:
            self.frame_mark()
            self.callback(self.screen_buffer)
            self.screen_buffer[:, :] = self.void_color

            self.t_pointer = self.frame_cntr*self.frame_period_us
            self.frame_cntr += 1
            self.process_data(events[frame.size:])
