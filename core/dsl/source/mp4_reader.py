import cv2

from core.event.subjects import OPENING, LATE_INIT, CLOSING, PIPELINE_READY
from core.dsl.source.module import Source


class Mp4Reader(Source):

    def __init__(self, mp4_path):
        super().__init__()

        self.capture = cv2.VideoCapture(mp4_path)
        self.frame_cnt = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.message_dispatcher.subscribe(PIPELINE_READY, self.iterate_frames)

    def iterate_frames(self):
        capture = self.capture
        properties = {
            'frame_cnt': self.frame_cnt,
            'fps': self.fps,
            'height': self.height,
            'width': self.width
        }

        self.message_dispatcher.notify(LATE_INIT, **properties)

        self.message_dispatcher.notify(OPENING)

        not_closed = True

        def stop():
            nonlocal not_closed
            not_closed = False

        self.message_dispatcher.subscribe(CLOSING, lambda: stop())

        while not_closed and capture.isOpened():
            ret, frame = capture.read()

            if not ret:
                break

            self.callback(frame, **properties)

        self.message_dispatcher.notify(CLOSING)
