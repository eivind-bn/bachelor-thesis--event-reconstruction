import numpy as np

from core.dsl.transformer.module import Transformer


class FrequencyFilter(Transformer):

    def __init__(self, low_bound=0.1, high_bound=0.9):
        super().__init__()
        assert 0.0 <= low_bound < high_bound <= 1.0

        self.low_bound = low_bound
        self.high_bound = high_bound

        self.last_timestamp = None

    def late_init(self, height, width, **kwargs):
        self.last_timestamp = np.zeros((height, width), dtype=np.int64)

    def process_data(self, events, **kwargs):

        yi, xi, ti = events['y'], events['x'], events['t']

        new_time = np.copy(self.last_timestamp)
        new_time[yi, xi] = ti

        frequency = 1 / (new_time[yi, xi] - self.last_timestamp[yi, xi])

        max_freq = np.max(frequency)
        min_freq = np.min(frequency)

        freq_norm = (frequency - min_freq)/(max_freq - min_freq)

        filtered = np.argwhere(np.logical_and(self.low_bound < freq_norm, freq_norm < self.high_bound))

        self.callback(events[filtered])

        self.last_timestamp = new_time


