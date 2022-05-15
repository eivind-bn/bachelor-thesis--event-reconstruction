from metavision_core.event_io import EventsIterator

from core.event.subjects import CLOSING, LATE_INIT, OPENING, PIPELINE_READY
from core.dsl.source.module import Source


class EventReader(Source):

    def __init__(self,
                 input_path: str,
                 start_ts=0,
                 mode='delta_t',
                 delta_t=10000,
                 n_events=10000,
                 max_duration=None,
                 relative_timestamps=False,
                 **kwargs):
        super().__init__()

        self.event_iterator = EventsIterator(
            input_path=input_path,
            start_ts=start_ts,
            mode=mode,
            delta_t=delta_t,
            n_events=n_events,
            max_duration=max_duration,
            relative_timestamps=relative_timestamps,
            **kwargs
        )

        self.message_dispatcher.subscribe(PIPELINE_READY, self.iterate_events)

    def iterate_events(self):
        event_iterator = self.event_iterator
        height, width = self.event_iterator.get_size()

        self.message_dispatcher.notify(LATE_INIT, height=height, width=width)

        self.message_dispatcher.notify(OPENING)

        closed = False

        def stop():
            nonlocal closed
            closed = True

        self.message_dispatcher.subscribe(CLOSING, lambda: stop())

        for events in event_iterator:
            if closed:
                break
            self.callback(events, height=height, width=width)

        self.message_dispatcher.notify(CLOSING)
