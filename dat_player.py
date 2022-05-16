import os

import cv2
import numpy as np
from metavision_core.event_io import DatWriter, EventsIterator
from numpy.core import records

from core.constants.colors import WHITE, BLUE, BLACK
from core.dsl.sink.event_writer import EventWriter
from core.dsl.source.events_reader import EventReader
from core.dsl.source.mp4_reader import Mp4Reader
from core.dsl.transformer.frames_by_event_batches import EventBatchToFrames
from core.dsl.transformer.mp4_to_singular_events import Mp4ToSingularEvents
from core.dsl.transformer.mp4_to_sliding_events import Mp4ToSlidingEvents
from core.dsl.transformer.mp4_to_stacking_events import Mp4ToStackingEvents


def main():
    import argparse

    parser = argparse.ArgumentParser(description='event-player',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i',
                        '--input-event-file',
                        dest='input_path',
                        required=True,
                        help="Path to event-file file")

    parser.add_argument('-d',
                        '--delta',
                        dest='delta_t',
                        required=True,
                        help="Path to mp4-video file")

    parser = parser.parse_args()

    input_file = parser.input_path
    delta_t = int(parser.delta_t)
    winname = os.path.basename(input_file)
    event_stream = EventsIterator(input_file, delta_t)
    height, width = event_stream.get_size()
    screen_buffer = np.zeros([height, width, 3], dtype=np.ubyte)

    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    print('To take a snapshot of the recording, press the \'c\' button.')

    def snapshot():
        print('Saving snapshot to:')
        print(os.path.abspath(os.curdir))
        cv2.imwrite('snapshot.png', screen_buffer)

    for events in event_stream:
        yi, xi, pi = events['y'], events['x'], events['p']

        ones = np.argwhere(pi == 1)
        zeros = np.argwhere(pi == 0)

        screen_buffer[yi[ones], xi[ones]] = WHITE
        screen_buffer[yi[zeros], xi[zeros]] = BLUE

        cv2.imshow(winname, screen_buffer)
        key = cv2.pollKey()

        if key != -1:
            key = chr(key)
            if key == 'c':
                snapshot()

        screen_buffer[:, :] = BLACK

        invisible = cv2.getWindowProperty(winname, cv2.WND_PROP_VISIBLE) == 0
        if invisible:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
