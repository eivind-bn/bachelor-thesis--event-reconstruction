import numpy as np
from numpy.core import records

from core.dsl.transformer.module import Transformer


class Mp4ToStackingEvents(Transformer):

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
            return np.empty((0,), dtype=self.event_dtype)

        timestamp = (self.frame_cnr / self.fps) * 1e6
        self.frame_cnr += 1

        new_levels = (np.mean(image / 255, axis=2) // self.threshold).astype(np.int64)

        pos_up = np.argwhere(new_levels > self.old_levels)
        pos_down = np.argwhere(new_levels < self.old_levels)

        grad_up = new_levels[pos_up[:, 0], pos_up[:, 1]]
        grad_down = new_levels[pos_down[:, 0], pos_down[:, 1]]

        pos_up = np.repeat(pos_up, grad_up, axis=0)
        pos_down = np.repeat(pos_down, grad_down, axis=0)

        polarity_up = np.ones((pos_up.shape[0], 1), dtype=np.int8)
        polarity_down = np.zeros((pos_down.shape[0], 1), dtype=np.int8)

        time = np.full((pos_up.shape[0] + pos_down.shape[0],), timestamp)

        pos_events = np.column_stack([pos_up, polarity_up])
        neg_events = np.column_stack([pos_down, polarity_down])

        events = np.row_stack([pos_events, neg_events])
        events = np.column_stack([events, time])

        self.old_levels = new_levels

        events = records.fromarrays(events.T, dtype=self.event_dtype)

        self.callback(events, **kwargs)


