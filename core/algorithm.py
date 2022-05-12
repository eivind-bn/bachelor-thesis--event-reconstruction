from abc import ABC, abstractmethod


class Algorithm(ABC):

    def __init__(self):
        self.callback = lambda _: {}

    @abstractmethod
    def process_data(self, data):
        pass

    def on_data_processed(self, callback):
        self.callback = callback
