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

        # First frame is the initial pixel states.
        if self.frame_cnr == 0:
            # Calculate the discrete pixel states.
            self.old_levels = (np.mean(image / 255, axis=2) // self.threshold).astype(np.int64)
            self.frame_cnr += 1
            return np.empty((0,), dtype=self.event_dtype)

        # Calculate timestamp shared among all events.
        timestamp = (self.frame_cnr / self.fps) * 1e6
        self.frame_cnr += 1

        # Calculate discrete states of iteratively.
        new_levels = (np.mean(image / 255, axis=2) // self.threshold).astype(np.int64)

        # Wherever an increase/decrease occurred, an event occurred.
        pos_up = np.argwhere(new_levels > self.old_levels)
        pos_down = np.argwhere(new_levels < self.old_levels)

        # Find the intensity difference.
        grad_up = new_levels[pos_up[:, 0], pos_up[:, 1]] - self.old_levels[pos_up[:, 0], pos_up[:, 1]]
        grad_down = new_levels[pos_down[:, 0], pos_down[:, 1]] - self.old_levels[pos_down[:, 0], pos_down[:, 1]]

        # Look at the discrete intensity change value, and make that many event duplicates.
        pos_up = np.repeat(pos_up, grad_up, axis=0)
        pos_down = np.repeat(pos_down, grad_down, axis=0)

        # Make polarity columns for events.
        polarity_up = np.ones((pos_up.shape[0], 1), dtype=np.int8)
        polarity_down = np.zeros((pos_down.shape[0], 1), dtype=np.int8)

        # Make time column for events. Each event has the same timestamp.
        time = np.full((pos_up.shape[0] + pos_down.shape[0],), timestamp)

        # Begin tabulating.
        pos_events = np.column_stack([pos_up, polarity_up])
        neg_events = np.column_stack([pos_down, polarity_down])

        # Join both event kinds to the same table.
        events = np.row_stack([pos_events, neg_events])
        events = np.column_stack([events, time])

        # New frame is old frame for next cycle.
        self.old_levels = new_levels

        # Make record which is how events are stored.
        events = records.fromarrays(events.T, dtype=self.event_dtype)

        # Done. Transfer to next module.
        self.callback(events, **kwargs)


