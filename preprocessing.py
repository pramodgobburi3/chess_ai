import cv2
import numpy as np
import os
import json
import re
import pandas as pd

BASEDIR = os.getcwd()

board_extraction_dir = os.path.join("data/board_extraction")
images_dir = os.path.join(board_extraction_dir, "images")
grayscale_images_dir = os.path.join(board_extraction_dir, "grayscale_images")
coordinates_dir = os.path.join(board_extraction_dir, "coordinates.json")

coordinates = {}
f_coordinates = open(coordinates_dir)
for line in f_coordinates:
    jline = json.loads(line)
    content = jline['content']
    points = jline['annotation'][0]["points"]
    if re.search("^.*resized_", content):
        [pre, suf] = re.split(r"^.*resized_", content)
        coordinates[suf] = points
    else:
        continue 

out = []
for file in os.listdir(images_dir):
    if file in coordinates:
        file_path = os.path.join(images_dir, file)
        out.append({
            'name': file,
            'coordinates': coordinates[file]
        })
        print(file_path)
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_path = os.path.join(grayscale_images_dir, file)
        cv2.imwrite(gray_path, gray)

df = pd.DataFrame(out)
df.to_csv("data.csv", index=False, sep=';')
# print(os.listdir(images_dir))