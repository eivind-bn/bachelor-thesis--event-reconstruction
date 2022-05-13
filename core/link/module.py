from abc import ABC


from core.sink.module import Sink
from core.source.module import Source


class Algorithm(Source, Sink, ABC):

    def __rshift__(self, consumer: Sink):
        self.on_data_processed(consumer.process_data)

    def __lshift__(self, provider: Source):
        provider.on_data_processed(self.process_data)
