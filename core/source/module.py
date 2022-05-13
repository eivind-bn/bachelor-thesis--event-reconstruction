from core.event.broadcaster import Broadcaster


class Source(object):

    def __init__(self):
        self.msg_dispatcher = Broadcaster()
        self.callback = lambda _: {}

    def on_data_processed(self, callback):
        self.callback = callback

    def __call__(self, callback):
        self.on_data_processed(callback.process_data)

    def __rshift__(self, consumer):
        self.on_data_processed(consumer.process_data)
