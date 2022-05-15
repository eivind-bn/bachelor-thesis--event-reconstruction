import numpy as np

from core.constants.colors import BLACK, WHITE, BLUE
from core.dsl.transformer.module import Transformer


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
        # Finding rows indicis where timestamp is less than the time-pointer.
        # These belong to the current frame.
        frame = np.argwhere(events['t'] < self.t_pointer)

        # Retrieving coordinates and polarity with the frame indicis.
        yi, xi, pi = events['y'][frame], events['x'][frame], events['p'][frame]

        # Finding indicis of events with different polarities.
        zeroes = np.argwhere(pi == 0)
        ones = np.argwhere(pi == 1)

        # Colorizing frame with distinct colors based on polarity.
        self.frame_buffer[yi[zeroes], xi[zeroes]] = self.neg_color
        self.frame_buffer[yi[ones], xi[ones]] = self.pos_color

        # Checking if there's remaining events not belonging to this frame.
        # If there is, then this frame is done.
        if frame.size < events.size:
            # Transferring complete frame.
            self.callback(self.frame_buffer, **kwargs)

            # Resting frame-buffer.
            self.frame_buffer[:, :] = self.void_color

            # Incrementing time-pointer.
            self.t_pointer = self.frame_cntr * self.frame_period_us
            self.frame_cntr += 1

            # Recursively process remaking events. Events are sorted with respect to time,
            # so serving the rest with slicing.
            self.process_data(events[frame.size:])
