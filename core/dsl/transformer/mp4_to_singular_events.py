import numpy as np
from numpy.core import records

from core.dsl.transformer.module import Transformer


class Mp4ToSingularEvents(Transformer):

    def __init__(self, threshold):
        super().__init__()

        self.threshold = threshold
        self.fps = None

        self.event_dtype = [('y', np.uint16), ('x', np.uint16), ('p', np.int16), ('t', np.int64)]
        self.old_levels = None
        self.frame_cnr = 0

    def late_init(self, fps, **kwargs):
        self.fps = fps

    def process_data(self, image, **kwargs):

        if self.frame_cnr == 0:
            self.old_levels = (np.mean(image / 255, axis=2) // self.threshold).astype(np.int64)
            self.frame_cnr += 1

        timestamp = (self.frame_cnr / self.fps) * 1e6
        self.frame_cnr += 1

        new_levels = (np.mean(image / 255, axis=2) // self.threshold).astype(np.int64)

        pos_cord = np.argwhere(new_levels > self.old_levels)
        dec_cord = np.argwhere(new_levels < self.old_levels)

        polarity_up = np.ones((pos_cord.shape[0], 1), dtype=np.int8)
        polarity_down = np.zeros((dec_cord.shape[0], 1), dtype=np.int8)

        time = np.full((pos_cord.shape[0] + dec_cord.shape[0],), timestamp)

        pos_events = np.column_stack([pos_cord, polarity_up])
        neg_events = np.column_stack([dec_cord, polarity_down])

        events = np.row_stack([pos_events, neg_events])
        events = np.column_stack([events, time])

        self.old_levels = new_levels

        events = records.fromarrays(events.T, dtype=self.event_dtype)

        self.callback(events, **kwargs)
