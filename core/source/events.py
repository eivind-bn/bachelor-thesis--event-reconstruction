from metavision_core.event_io import EventsIterator

from core.source.module import Source


class EventStream(Source):

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

    def on_data_processed(self, callback):
        event_iterator = self.event_iterator

        is_lambda = isinstance(callback, type(lambda: 0))
        if is_lambda:
            arg_count = callback.__code__.co_argcount
        else:
            arg_count = callback.__call__.__code__.co_argcount - 1

        if arg_count == 3:
            height, width = self.event_iterator.get_size()

            for events in event_iterator:
                callback(height, width, events)

        elif arg_count == 1:

            for events in event_iterator:
                callback(events)
