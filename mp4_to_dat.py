import os

from core.dsl.sink.event_writer import EventWriter
from core.dsl.source.mp4_reader import Mp4Reader
from core.dsl.transformer.mp4_to_singular_events import Mp4ToSingularEvents
from core.dsl.transformer.mp4_to_sliding_events import Mp4ToSlidingEvents
from core.dsl.transformer.mp4_to_stacking_events import Mp4ToStackingEvents


def main():
    import argparse

    def raise_error(error):
        raise error

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
                        required=False,
                        help="Path to dat-event file")

    parser.add_argument('-em',
                        '--event-mode',
                        dest='event_mode',
                        required=False,
                        default='singular',
                        choices=['singular', 'sliding', 'stacking'],
                        help="Determines how event thresholds shall behave.")

    parser.add_argument('-et',
                        '--event-threshold',
                        dest='threshold',
                        default='0.1',
                        type=lambda t: float(t) if 0.0 < float(t) < 1.0 else raise_error(ValueError),
                        help='''
                        Threshold which greyscale pixel gradients must exceed to create an event.
                        Number is a percent factor which signifies percent of pixel brightness.
                        For example, 0.1 means a pixel must change 10% of maximum of pixel brightness to
                        fire an event.''')

    parser = parser.parse_args()

    event_mode = parser.event_mode

    input_path: str = parser.input_path
    assert input_path.endswith('.mp4')

    output_path = parser.output_path
    if output_path is None:
        input_filename = os.path.basename(input_path)
        input_dir = os.path.dirname(input_path)
        output_path = f'{input_dir}/../decon/{input_filename[:-4]}.dat'
    else:
        assert output_path.endswith('.dat')

    threshold = parser.threshold
    mp4_in = Mp4Reader(input_path)
    dat_out = EventWriter(output_path)

    def mp4_to_dat():
        i = 1

        def progress_report(data, **kwargs):
            nonlocal i
            n_frame = kwargs.get('frame_cnt', '?')

            print(f'{i}/{n_frame} frames.', end='\r')
            i += 1

        if event_mode == 'singular':
            mp4_in >> Mp4ToSingularEvents(threshold) >> progress_report >> dat_out

        elif event_mode == 'stacking':
            mp4_in >> Mp4ToStackingEvents(threshold) >> progress_report >> dat_out

        elif event_mode == 'sliding':
            mp4_in >> Mp4ToSlidingEvents(threshold) >> progress_report >> dat_out

    mp4_to_dat()


if __name__ == "__main__":
    main()
