
from core.event.event_chain import EventDispatcher
from core.event.subjects import BROADCASTER_JOINED


class Source(EventDispatcher):

    def __init__(self):
        super().__init__()
        self.callback = lambda data, **kwargs:None

    def on_data_processed(self, consumer):
        old_cb = self.callback
        new_cb = consumer

        def new_callback(data, **kwargs):
            old_cb(data, **kwargs)
            new_cb(data, **kwargs)

        self.callback = new_callback

        if issubclass(type(consumer), EventDispatcher):
            self.msg_dispatcher.join(consumer.msg_dispatcher)
            self.msg_dispatcher.notify(BROADCASTER_JOINED)
            return consumer
        else:
            return self

    def __rshift__(self, consumer):
        return self.on_data_processed(consumer)

