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

        # First frame is initial state.
        if self.frame_cnr == 0:
            self.old_levels = (np.mean(image / 255, axis=2) // self.threshold).astype(np.int64)
            self.frame_cnr += 1

        # Timestamp for every event present in this frame.
        timestamp = (self.frame_cnr / self.fps) * 1e6
        self.frame_cnr += 1

        # Calculating normalized greyscale with average method.
        new_levels = (np.mean(image / 255, axis=2) // self.threshold).astype(np.int64)

        # Finding coordinates which has increased/decreased.
        pos_cord = np.argwhere(new_levels > self.old_levels)
        dec_cord = np.argwhere(new_levels < self.old_levels)

        # Making polarity columns for both events.
        polarity_up = np.ones((pos_cord.shape[0], 1), dtype=np.int8)
        polarity_down = np.zeros((dec_cord.shape[0], 1), dtype=np.int8)

        # Making time column shared among all events.
        time = np.full((pos_cord.shape[0] + dec_cord.shape[0],), timestamp)

        # Joining to a table.
        pos_events = np.column_stack([pos_cord, polarity_up])
        neg_events = np.column_stack([dec_cord, polarity_down])

        # Joining both events to the same table.
        events = np.row_stack([pos_events, neg_events])
        events = np.column_stack([events, time])

        # Setting new-state to old-state for next cycle.
        self.old_levels = new_levels

        # Making record which is how events typically are stored.
        events = records.fromarrays(events.T, dtype=self.event_dtype)

        # Done. Sending to consumer.
        self.callback(events, **kwargs)
