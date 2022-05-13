from abc import ABC, abstractmethod

from core.event.broadcaster import Broadcaster


class Sink(ABC, object):

    def __init__(self):
        self.msg_dispatcher = Broadcaster()

    @abstractmethod
    def process_data(self, data, **kwargs):
        pass

    def __call__(self, data, **kwargs):
        self.process_data(data, **kwargs)

    def __lshift__(self, provider):
        provider.on_data_processed(self.process_data)
