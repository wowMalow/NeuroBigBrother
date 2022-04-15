import time
import os
from itertools import count

SESSION_ID = int(time.time())

image_counter = count(1, 1)

LEFT_EYE = (33, 133)
RIGHT_EYE = (362, 263)

LEFT_IRIS = (469, 471)
RIGHT_IRIS = (474, 476)

os.makedirs(f'./data/{SESSION_ID}')
