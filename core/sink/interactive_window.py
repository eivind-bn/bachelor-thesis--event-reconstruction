import sys

import cv2
import numpy as np
from metavision_sdk_ui import Window, MTWindow, BaseWindow, UIAction, UIKeyEvent, EventLoop

from core.sink.module import Sink
import cv2


class InteractiveWindow(Sink):

    def __init__(self,
                 height,
                 width,
                 title,
                 color_palette=BaseWindow.RenderMode.BGR):

        self.title = title
        self.cursor_log = [[0, 0]] * 10
        self.is_paused = False
        self.color_palette = color_palette
        cv2.namedWindow('image', cv2.WINDOW_FREERATIO)

    def process_data(self, height, width, data):
        is_visible = cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) == 1
        if not is_visible:
            quit(0)

        t = np.zeros((height, width, 3), dtype=np.ubyte)

        ones = np.argwhere(data['p'] == 1)
        zeroes = np.argwhere(data['p'] == 0)

        t[data['y'][ones], data['x'][ones]] = [255,255,255]
        t[data['y'][zeroes], data['x'][zeroes]] = [255, 255, 255]

        cv2.imshow('image', t)

        key = cv2.pollKey()
        if key != -1:
            print(key)
        # cv2.waitKey(1000)


