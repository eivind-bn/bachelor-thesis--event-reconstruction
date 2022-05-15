import numpy as np

from core.dsl.transformer.module import Transformer


class MP4ToGreyscale(Transformer):

    def process_data(self, image, **kwargs):
        # Calculating greyscale with average-method.
        greyscale = np.mean(image, axis=2).astype(np.ubyte)

        # Need to repeat each subpixel to form a pixel.
        new_frame = np.repeat(greyscale[:, :, np.newaxis], 3, axis=2)

        # Done.
        self.callback(new_frame)
