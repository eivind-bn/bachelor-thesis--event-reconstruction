import os

import numpy as np
import cv2
from metavision_core.event_io import DatWriter, EventsIterator
from numpy.core import records


def mk_sliding_threshold_events(mp4_path, dat_path, threshold=0.1):
    cap = cv2.VideoCapture(mp4_path)

    global old_states

    event_dtype = [('y', np.uint16), ('x', np.uint16), ('p', np.int16), ('t', np.int64)]
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    dat_writer = DatWriter(dat_path, height, width)

    if cap.isOpened():
        ret, frame = cap.read()
        old_states = np.mean(frame / 255, axis=2)

    i = 1
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        print(f'frame: {i}/{frame_cnt}', end='\r')
        t = int((i / fps) * 10e6)
        i += 1

        new_states = np.mean(frame / 255, axis=2)

        delta_up = new_states >= (old_states + threshold)
        delta_down = new_states <= (old_states - threshold)
        new_events = delta_up | delta_down

        old_states = np.where(new_events, new_states, old_states)

        cord_up = np.argwhere(delta_up)
        cord_down = np.argwhere(delta_down)

        pos_events = np.column_stack([cord_up, np.ones((cord_up.shape[0],), dtype=np.int16)])
        neg_events = np.column_stack([cord_down, np.zeros((cord_down.shape[0],), dtype=np.int16)])

        events = np.row_stack([pos_events, neg_events])
        events = np.column_stack([events, np.full((events.shape[0],), t, dtype=np.int64)])
        events = records.fromarrays(events.T, dtype=event_dtype)

        dat_writer.write(events)


def mp4_to_dat_v2(mp4_path, dat_path, threshold=0.1):
    cap = cv2.VideoCapture(mp4_path)

    global old_levels

    event_dtype = [('y', np.uint16), ('x', np.uint16), ('p', np.int16), ('t', np.int64)]
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    dat_writer = DatWriter(dat_path, height, width)

    if cap.isOpened():
        ret, frame = cap.read()
        old_levels = (np.mean(frame / 255, axis=2) // threshold).astype(np.int64)

    timestamps = ((np.arange(frame_cnt + 1) / fps) * 10e6).astype(np.int64)
    for i in range(frame_cnt):
        ret, frame = cap.read()
        time_a, time_b = timestamps[i:i + 2]

        print(f'frame: {i}/{frame_cnt}', end='\r')

        new_levels = (np.mean(frame / 255, axis=2) // threshold).astype(np.int64)

        pos_up = np.argwhere(new_levels > old_levels)
        pos_down = np.argwhere(new_levels < old_levels)

        grad_up = new_levels[pos_up[:, 0], pos_up[:, 1]]
        grad_down = new_levels[pos_down[:, 0], pos_down[:, 1]]

        pos_up = np.repeat(pos_up, grad_up, axis=0)
        pos_down = np.repeat(pos_down, grad_down, axis=0)

        polarity_up = np.ones((pos_up.shape[0], 1), dtype=np.int8)
        polarity_down = np.zeros((pos_down.shape[0], 1), dtype=np.int8)

        time = np.full((pos_up.shape[0] + pos_down.shape[0],), time_b)

        pos_events = np.column_stack([pos_up, polarity_up])
        neg_events = np.column_stack([pos_down, polarity_down])

        events = np.row_stack([pos_events, neg_events])
        events = np.column_stack([events, time])

        dat_writer.write(records.fromarrays(events.T, dtype=event_dtype))

        old_levels = new_levels


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='mp4-to-dat converter',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i',
                        '--input-mp4-file',
                        dest='input_path',
                        required=True,
                        help="Path to mp4-video file")

    parser.add_argument('-o',
                        '--output-dat-file',
                        dest='output_path',
                        required=True,
                        help="Path to dat-event file")

    parser.add_argument('-o',
                        '--output-dat-file',
                        dest='output_path',
                        required=True,
                        help="Path to dat-event file")


    return parser.parse_args()


def main():
    """ Main """
    args = parse_args()

    if not os.path.isfile(args.input_path):
        print('Fail to access RAW file ' + args.input_path)
        return

    # Events iterator on Camera or RAW file
    mv_iterator = EventsIterator(input_path=args.input_path, delta_t=1000)

    with open('cd.csv', 'w') as csv_file:

        # Read formatted CD events from EventsIterator & write to CSV
        for evs in mv_iterator:
            for (x, y, p, t) in evs:
                csv_file.write("%d,%d,%d,%d\n" % (x, y, p, t))


if __name__ == "__main__":
    main()