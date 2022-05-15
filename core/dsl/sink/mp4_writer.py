import cv2

from core.dsl.sink.module import Sink
from core.event.subjects import CLOSING, BROADCASTER_JOINED, PIPELINE_READY


class Mp4Writer(Sink):

    def __init__(self, mp4_path, fps):
        super().__init__()

        assert mp4_path.endswith('.mp4')

        self.mp4_path = mp4_path
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')

        self.mp4_writer = None
        self.height = None
        self.width = None

        self.message_dispatcher.subscribe(CLOSING, lambda: self.mp4_writer.release())
        self.message_dispatcher.subscribe(BROADCASTER_JOINED, lambda: self.message_dispatcher.notify(PIPELINE_READY))

    def late_init(self, height, width, **kwargs):
        self.height = height
        self.width = width
        self.mp4_writer = cv2.VideoWriter(self.mp4_path, self.fourcc, self.fps, (width, height))

    def process_data(self, data, **kwargs):
        self.mp4_writer.write(data)
