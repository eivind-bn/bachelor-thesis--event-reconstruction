import numpy as np
from scipy.ndimage import gaussian_filter

from core.constants.colors import WHITE
from core.dsl.transformer.module import Transformer


class AsymptoticIntensityPredictor(Transformer):

    def __init__(self,
                 gaussian_filter_sigma=1.0,
                 colorspace=225,
                 colorspace_offset=0,
                 intensity_decay=0.2,
                 intensity_impedance=1.0):
        super().__init__()

        self.sigma = gaussian_filter_sigma
        self.colorspace = colorspace
        self.colorspace_offset = colorspace_offset
        self.intensity_decay = intensity_decay
        self.intensity_impedance = intensity_impedance

        self.screen_buffer = None
        self.intensity_inference = None

    def late_init(self, height, width, **kwargs):
        self.screen_buffer = np.full([height, width, 3], WHITE, dtype=np.ubyte)
        self.intensity_inference = np.zeros((height, width), dtype=np.float64)

    def process_data(self, events, **kwargs):
        delta_up = np.argwhere(events['p'] == 1)
        delta_down = np.argwhere(events['p'] == 0)

        decay = self.intensity_decay * self.intensity_inference
        decay[events['y'], events['x']] = 0
        self.intensity_inference -= decay

        temp = self.intensity_inference[events['y'], events['x']]
        self.intensity_inference = gaussian_filter(self.intensity_inference, sigma=self.sigma)
        self.intensity_inference[events['y'], events['x']] = temp

        self.intensity_inference[events['y'][delta_up], [events['x'][delta_up]]] += \
            self.intensity_impedance * (
                        (1 - self.intensity_inference[events['y'][delta_up], [events['x'][delta_up]]]) / 2)

        self.intensity_inference[events['y'][delta_down], [events['x'][delta_down]]] += \
            self.intensity_impedance * (
                        (-1 - self.intensity_inference[events['y'][delta_down], [events['x'][delta_down]]]) / 2)

        greyscale_values = ((self.intensity_inference + 1) * (self.colorspace / 2) + self.colorspace_offset).astype(
            np.ubyte)

        self.screen_buffer[:, :] = np.repeat(greyscale_values[:, :, np.newaxis], 3, axis=2)
        self.callback(self.screen_buffer, **kwargs)