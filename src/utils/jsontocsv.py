# -*- coding: utf-8 -*-
import glob
import json

import pandas as pd

files = glob.glob("src/data/raw/" + "*.json")
if not files:
    raise Exception("No JSON files found!")

df = pd.DataFrame(
    columns=["title", "review_text_raw", "Grafik", "Sound", "Steuerung", "Atmosphäre"]
)

for file in files:
    with open(file, "r") as f:
        json_f = json.load(f)
        title = file.split("/")[-1].split("\\")[-1].split(".")[0]

        for review in json_f["reviews"]:
            if len(review["rating"]) > 0:

                tmp_review = {
                    "title": title,
                    "review_text_raw": review["text"],
                    "Grafik": review["rating"]["Grafik"],
                    "Sound": review["rating"]["Sound"],
                    "Steuerung": review["rating"]["Steuerung"],
                    "Atmosphäre": review["rating"]["Atmosphäre"],
                }
                df = df.append(tmp_review, ignore_index=True)

df.to_csv("src/data/data_raw.csv", index=False)
