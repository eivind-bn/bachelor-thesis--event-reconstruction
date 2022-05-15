from metavision_core.event_io import DatWriter

from core.dsl.sink.module import Sink
from core.event.subjects import CLOSING, BROADCASTER_JOINED, PIPELINE_READY


class EventWriter(Sink):

    def __init__(self, dat_path):
        super().__init__()

        self.dat_path = dat_path
        self.dat_writer = None
        self.height = None
        self.width = None

        self.message_dispatcher.subscribe(CLOSING, lambda: self.dat_writer.close())
        self.message_dispatcher.subscribe(BROADCASTER_JOINED, lambda: self.message_dispatcher.notify(PIPELINE_READY))

    def late_init(self, height, width, **kwargs):
        self.dat_writer = DatWriter(self.dat_path, height, width)

    def process_data(self, events, **kwargs):
        self.dat_writer.write(events)
