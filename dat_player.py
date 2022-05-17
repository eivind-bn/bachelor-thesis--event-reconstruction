import os

import cv2
import numpy as np
from metavision_core.event_io import DatWriter, EventsIterator
from numpy.core import records

from core.constants.colors import WHITE, BLUE, BLACK
from core.dsl.sink.event_writer import EventWriter
from core.dsl.source.events_reader import EventReader
from core.dsl.source.mp4_reader import Mp4Reader
from core.dsl.transformer.event_to_intensity_predictor import AsymptoticIntensityPredictor
from core.dsl.transformer.frames_by_event_batches import EventBatchToFrames
from core.dsl.transformer.mp4_to_singular_events import Mp4ToSingularEvents
from core.dsl.transformer.mp4_to_sliding_events import Mp4ToSlidingEvents
from core.dsl.transformer.mp4_to_stacking_events import Mp4ToStackingEvents


def snapshot(**images):
    print('Saving snapshot to:')
    print(f'{os.path.abspath(os.curdir)}/snapshots')
    for name, image in images.items():
        cv2.imwrite(f'snapshots/{name}', image)
        print(f'Saved: {os.path.abspath(os.curdir)}/snapshots/{name}')


def raw(parser):
    input_file = parser.input_path
    delta_t = int(parser.delta_t)
    filename = os.path.basename(input_file)
    event_stream = EventsIterator(input_file, delta_t=delta_t, start_ts=0)
    height, width = event_stream.get_size()
    screen_buffer = np.zeros([height, width, 3], dtype=np.ubyte)

    cv2.namedWindow(filename, cv2.WINDOW_FREERATIO)
    print('To take a snapshot of the recording, press the \'c\' button.')

    for events in event_stream:
        if events.size < 1:
            continue

        yi, xi, pi = events['y'], events['x'], events['p']

        ones = np.argwhere(pi == 1)
        zeros = np.argwhere(pi == 0)

        screen_buffer[yi[ones], xi[ones]] = WHITE
        screen_buffer[yi[zeros], xi[zeros]] = BLUE

        cv2.imshow(filename, screen_buffer)
        key = cv2.pollKey()

        if key != -1:
            key = chr(key)
            if key == 'c':
                snapshot(**{filename[:-4]: screen_buffer})

        screen_buffer[:, :] = BLACK

        invisible = cv2.getWindowProperty(filename, cv2.WND_PROP_VISIBLE) == 0
        if invisible:
            break

    cv2.destroyAllWindows()


def to_frame(events, height, width):
    screen_buffer = np.zeros([height, width, 3], dtype=np.ubyte)
    yi, xi, pi = events['y'], events['x'], events['p']

    ones = np.argwhere(pi == 1)
    zeros = np.argwhere(pi == 0)

    screen_buffer[yi[ones], xi[ones]] = WHITE
    screen_buffer[yi[zeros], xi[zeros]] = BLUE

    return screen_buffer


def reconstruct(parser):
    combo = parser.render_combo
    input_file = parser.input_path
    delta_t = int(parser.delta_t)
    gaussian_filter = parser.gaussian_filter
    intensity_decay = parser.intensity_decay
    filename = os.path.basename(input_file)
    event_stream = EventsIterator(input_file, delta_t=delta_t, start_ts=0)
    height, width = event_stream.get_size()
    reconstructor = AsymptoticIntensityPredictor(gaussian_filter, intensity_decay)
    reconstructor.late_init(height, width)
    reconstruction = np.zeros([height, width, 3], dtype=np.ubyte)

    cv2.namedWindow(filename, cv2.WINDOW_FREERATIO)
    print('To take a snapshot of the recording, press the \'c\' button.')

    def render(data, **kwargs):
        nonlocal reconstruction
        reconstruction = data
        cv2.imshow(filename, data)

    def render_combo(data, raw, **kwargs):
        nonlocal reconstruction
        reconstruction = data
        combo_frame = np.hstack([data, to_frame(raw, height, width)])
        cv2.imshow(filename, combo_frame)

    if combo:
        reconstructor.on_data_processed(render_combo)
    else:
        reconstructor.on_data_processed(render)

    for events in event_stream:
        if events.size < 1:
            continue

        reconstructor.process_data(events, raw=events)
        key = cv2.pollKey()

        if key != -1:
            key = chr(key)
            if key == 'c':
                snapshot(**{
                    f'{filename[:-4]}-raw.png': to_frame(events, height, width),
                    f'{filename[:-4]}-recon.png': reconstruction
                })

        invisible = cv2.getWindowProperty(filename, cv2.WND_PROP_VISIBLE) == 0
        if invisible:
            break

    cv2.destroyAllWindows()


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

    parser.add_argument('-r',
                        '--reconstruct',
                        dest='reconstruct',
                        type=bool,
                        nargs='?',
                        const=True,
                        default=False,
                        help='Flag to apply reconstruction')

    parser.add_argument('-ga',
                        '--gauss',
                        dest='gaussian_filter',
                        type=float,
                        default=0.3,
                        help='Gaussian filter sigma for the reconstruction')

    parser.add_argument('-de',
                        '--decay',
                        dest='intensity_decay',
                        type=float,
                        default=0.05,
                        help='Intensity decay on pixel for reconstruction')

    parser.add_argument('-c',
                        '--combo',
                        dest='render_combo',
                        type=bool,
                        nargs='?',
                        const=True,
                        default=False,
                        help='Render both raw a reconstruction at the same window')

    parser = parser.parse_args()

    if parser.reconstruct:
        reconstruct(parser)
    else:
        raw(parser)


if __name__ == "__main__":
    main()
