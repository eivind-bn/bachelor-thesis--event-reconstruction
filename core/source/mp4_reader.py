import cv2

from core.event.subjects import OPENING, LATE_INIT, CLOSING
from core.source.module import Source


class Mp4Reader(Source):

    def __init__(self, mp4_path):
        super().__init__()

        self.capture = cv2.VideoCapture(mp4_path)
        self.frame_cnt = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.msg_dispatcher.subscribe(OPENING, self.iterate_frames)

    def iterate_frames(self):
        capture = self.capture
        properties = {
            'capture': self.capture,
            'frame_cnt': self.frame_cnt,
            'fps': self.fps,
            'height': self.height,
            'width': self.width
        }

        self.msg_dispatcher.notify(LATE_INIT, **properties)
        while capture.isOpened():
            ret, frame = capture.read()

            if not ret:
                break

            self.callback(frame)

        self.msg_dispatcher.notify(CLOSING)
