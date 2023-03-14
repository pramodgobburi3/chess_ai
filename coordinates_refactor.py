import os
import json

BASEDIR = os.getcwd()

board_extraction_dir = os.path.join("data/board_extraction")

f_name = os.path.join(board_extraction_dir, "coordinates.json")

f = open(f_name)

for line in f:
    print(line)

