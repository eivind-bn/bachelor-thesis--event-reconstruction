import pandas as pd

bgr_colors = pd.read_csv('core\\constants\\colours_rgb_shades.csv')['R;G;B Dec']\
    .map(lambda x: x.split(';'))\
    .map(lambda x: [int(x[2]), int(x[1]), int(x[0])])\
    .tolist()

BLACK = [0, 0, 0]
RED = [0, 0, 255]
GREEN = [0, 255, 0]
YELLOW = [0, 255, 255]
BLUE = [255, 0, 0]
MAGENTA = [255, 0, 255]
CYAN = [255, 255, 0]
WHITE = [255, 255, 255]

colors = {
    'black': [0, 0, 0],
    'red': [0, 0, 255],
    'green': [0, 255, 0],
    'yellow': [0, 255, 255],
    'blue': [255, 0, 0],
    'magenta': [255, 0, 255],
    'cyan': [255, 255, 0],
    'white': [255, 255, 255]
}
