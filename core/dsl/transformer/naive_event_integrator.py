import numpy as np
from scipy.ndimage import gaussian_filter

from core.constants.colors import WHITE, BLACK
from core.dsl.transformer.module import Transformer


class SpanningIntensityPredictor(Transformer):

    def __init__(self, sigma):
        super().__init__()

        self.sigma = sigma

        self.screen_buffer = None
        self.state_cnt = None

    def late_init(self, height, width, **kwargs):
        self.screen_buffer = np.zeros([height, width, 3], dtype=np.ubyte)
        self.state_cnt = np.zeros((height, width), dtype=np.int64)

    def process_data(self, events, **kwargs):
        gaussian_filter(self.state_cnt, self.sigma)

        pos_events = np.argwhere(events['p'] == 1)
        neg_events = np.argwhere(events['p'] == 0)

        y_pos, x_pos = events['y'][pos_events], events['x'][pos_events]
        y_neg, x_neg = events['y'][neg_events], events['x'][neg_events]

        self.state_cnt[y_pos, x_pos] += 1
        self.state_cnt[y_neg, x_neg] -= 1

        min_cnt = np.min(self.state_cnt)
        max_cnt = np.max(self.state_cnt)

        greyscale = (((self.state_cnt - min_cnt) / (max_cnt - min_cnt)) * 255).astype(np.ubyte)


        self.screen_buffer[:, :] = np.repeat(greyscale[:, :, np.newaxis], 3, axis=2)
        self.callback(self.screen_buffer)

        self.screen_buffer[:, :] = BLACK


