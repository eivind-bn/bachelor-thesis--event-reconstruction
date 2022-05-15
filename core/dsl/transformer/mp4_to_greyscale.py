import numpy as np

from core.dsl.transformer.module import Transformer


class MP4ToGreyscale(Transformer):

    def process_data(self, image, **kwargs):
        greyscale = np.mean(image, axis=2).astype(np.ubyte)
        new_frame = np.repeat(greyscale[:, :, np.newaxis], 3, axis=2)
        self.callback(new_frame)
