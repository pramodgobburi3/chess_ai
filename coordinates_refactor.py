import os
import json

BASEDIR = os.getcwd()

board_extraction_dir = os.path.join("data/board_extraction")

f_name = os.path.join(board_extraction_dir, "coordinates.json")

f_out_name = os.path.join(board_extraction_dir, "coordinates_refactored.json")

f = open(f_name)
f_out = open(f_name, "w")

for line in f:
    print(line)

