
from core.event.event_chain import EventDispatcher
from core.event.subjects import BROADCASTER_JOINED, CLOSING


class Source(EventDispatcher):

    def __init__(self):
        super().__init__()
        self.callback = lambda data, **kwargs: None

    def on_data_processed(self, consumer):
        old_cb = self.callback
        new_cb = consumer

        def new_callback(data, **kwargs):
            try:
                old_cb(data, **kwargs)
                new_cb(data, **kwargs)
            except Exception:
                self.message_dispatcher.notify(CLOSING)
                raise

        self.callback = new_callback

        if issubclass(type(consumer), EventDispatcher):
            self.message_dispatcher.join(consumer.message_dispatcher)
            self.message_dispatcher.notify(BROADCASTER_JOINED)
            return consumer
        else:
            return self

    def __rshift__(self, consumer):
        return self.on_data_processed(consumer)

