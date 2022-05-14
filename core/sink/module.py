from abc import ABC, abstractmethod

from core.event.event_chain import EventDispatcher
from core.event.subjects import LATE_INIT


class Sink(ABC, EventDispatcher):

    def __init__(self):
        super().__init__()
        self.msg_dispatcher.subscribe(LATE_INIT, self.handle_late_init)

    def handle_late_init(self, **kwargs):
        self.late_init(**kwargs)
        self.msg_dispatcher.unsubscribe(LATE_INIT, self.handle_late_init)

    def late_init(self, **kwargs):
        pass

    @abstractmethod
    def process_data(self, data, **kwargs):
        pass

    def __call__(self, data, **kwargs):
        self.process_data(data, **kwargs)
