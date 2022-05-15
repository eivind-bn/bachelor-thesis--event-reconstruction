import cv2

from core.constants.colors import BLACK, WHITE, BLUE
from core.event.subjects import CLOSING, OPENING, BROADCASTER_JOINED, PIPELINE_READY
from core.dsl.sink.module import Sink


class Window(Sink):

    def __init__(self,
                 title,
                 void_color=BLACK,
                 pos_color=WHITE,
                 neg_color=BLUE):

        super().__init__()

        self.height = None
        self.width = None

        self.title = title
        self.void_color = void_color
        self.pos_color = pos_color
        self.neg_color = neg_color

        self.message_dispatcher.subscribe(OPENING, lambda: cv2.namedWindow(title))
        self.message_dispatcher.subscribe(CLOSING, lambda: self.close_window())
        self.message_dispatcher.subscribe(BROADCASTER_JOINED, lambda: self.message_dispatcher.notify(PIPELINE_READY))

    def late_init(self, height, width, **kwargs):
        self.height = height
        self.width = width

    def close_window(self):
        try:
            cv2.destroyWindow(self.title)
        except cv2.error:
            pass

    def process_data(self, image, **kwargs):
        if not image.shape == (self.height, self.width, 3):
            self.message_dispatcher.notify(CLOSING)
            raise TypeError

        is_visible = cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) == 1
        if not is_visible:
            self.message_dispatcher.notify(CLOSING)
            return

        cv2.imshow(self.title, image)

        keypress = cv2.pollKey()
        if keypress != -1:
            self.message_dispatcher.notify('keypress', key=chr(keypress))
