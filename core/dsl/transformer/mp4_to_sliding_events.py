import numpy as np
from numpy.core import records

from core.dsl.transformer.module import Transformer


class Mp4ToSlidingEvents(Transformer):

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

        # Set first frame greyscale as initial state and return early.
        # Event requires two frames to be calculated.
        if self.frame_cnr == 0:
            self.old_levels = (np.mean(image / 255, axis=2) // self.threshold).astype(np.int64)
            self.frame_cnr += 1
            return np.empty((0,), dtype=self.event_dtype)

        timestamp = (self.frame_cnr / self.fps) * 1e6
        self.frame_cnr += 1

        # Find greyscale of current frame using the average method.
        new_states = np.mean(image / 255, axis=2)

        # Finding pixels which has increased or decreased beyond the threshold of its previous state.
        delta_up = new_states >= (self.old_levels + self.threshold)
        delta_down = new_states <= (self.old_levels - self.threshold)

        # If increased/decreased, an event occurred, and is registered here.
        new_events = delta_up | delta_down

        # Where an event occurred, the new state is registered as old state for the next iteration.
        # Conversely, the old state is retained.
        self.old_levels = np.where(new_events, new_states, self.old_levels)

        # Yielding indicis by finding true-conditions in supplied boolean-array.
        cord_up = np.argwhere(delta_up)
        cord_down = np.argwhere(delta_down)

        # Tabulating coordinates with its respective event polarity.
        pos_events = np.column_stack([cord_up, np.ones((cord_up.shape[0],), dtype=np.int16)])
        neg_events = np.column_stack([cord_down, np.zeros((cord_down.shape[0],), dtype=np.int16)])

        # Appending both polarities in same table.
        events = np.row_stack([pos_events, neg_events])

        # Adding the time column. All events from this iteration share the same timestamp.
        events = np.column_stack([events, np.full((events.shape[0],), timestamp, dtype=np.int64)])

        # Converting array to record which is default Metavision format.
        events = records.fromarrays(events.T, dtype=self.event_dtype)

        # Dispatching events to the next consumer.
        self.callback(events, **kwargs)
