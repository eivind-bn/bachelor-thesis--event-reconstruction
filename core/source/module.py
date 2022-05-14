
from core.event.event_chain import EventDispatcher
from core.event.subjects import BROADCASTER_JOINED


class Source(EventDispatcher):

    def __init__(self):
        super().__init__()
        self.callback = None

    def on_data_processed(self, consumer):
        self.callback = consumer

        if issubclass(type(consumer), EventDispatcher):
            self.msg_dispatcher.join(consumer.msg_dispatcher)
            self.msg_dispatcher.notify(BROADCASTER_JOINED)
            return consumer
        else:
            return self

    def __rshift__(self, consumer):
        return self.on_data_processed(consumer)

