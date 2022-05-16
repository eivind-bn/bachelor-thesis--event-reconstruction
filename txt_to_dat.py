import numpy as np
import pandas as pd
from metavision_core.event_io import DatWriter


def txt_to_dat(txt_path, dat_path):
    EVENT_DTYPE = [('x', np.uint16), ('y', np.uint16), ('p', np.int16), ('t', np.int64)]
    with open(txt_path) as txt:
        width, height = txt.readline().split(' ')
        dat_writer = DatWriter(dat_path, int(height), int(width))

        df = pd.read_csv(txt, names=['t', 'x', 'y', 'p'], sep=' ')
        df['t'] *= 10e6
        df['t'] -= df['t'][0]
        df = df.to_records(index=False)[['x', 'y', 'p', 't']].astype(EVENT_DTYPE)
        df['y'] = int(height) - 1 - df['y']
        df['x'] = int(width) - 1 - df['x']
        dat_writer.write(df)
        dat_writer.close()

if __name__ == '__main__':
    import argparse


    def raise_error(error):
        raise error


    parser = argparse.ArgumentParser(description='txt-to-dat converter',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i',
                        '--input-txt-file',
                        dest='input_path',
                        required=True,
                        help="Path to txt-events file")

    parser.add_argument('-o',
                        '--output-dat-file',
                        dest='output_path',
                        required=True,
                        help="Path to dat-event file")

    parser = parser.parse_args()

    input_path = parser.input_path
    output_path = parser.output_path

    txt_to_dat(input_path, output_path)
