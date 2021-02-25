import json
from collections import Counter

import numpy as np
import pandas as pd

if __name__ == "__main__":

    # load data
    df_tokens = pd.read_csv("src/data/data_aspects_tokens.csv")
    df_tokens.drop_duplicates(inplace=True, ignore_index=True)
    df_raw = pd.read_csv("src/data/data_raw.csv")

    df_tokens["sentiment_words"] = df_tokens["sentiment_words"].str.replace("'", '"')
    df_tokens["sentiment_words"] = df_tokens["sentiment_words"].apply(
        lambda x: json.loads(x) if (len(x) > 2) else []
    )

    df_tokens["titel"] = df_raw["titel"][
        df_tokens["reviewnumber"].astype(int).tolist()
    ].tolist()

    # count how often people reference an aspect for a game
    output = pd.DataFrame()
    output["total_aspects"] = (
        df_tokens[["titel", "aspect"]]
        .groupby("titel")["aspect"]
        .apply(list)
        .apply(
            lambda x: [
                x.count(aspect)
                for aspect in ["Grafik", "Sound", "Steuerung", "Atmosphäre"]
            ]
        )
    )

    # get the top 3 most common words used to describe a game
    output["aspect_words"] = (
        df_tokens[["titel", "sentiment_words"]]
        .groupby("titel")["sentiment_words"]
        .apply(lambda x: [i[0] for i in x if len(i) > 0])
    )
    # output['aspect_words'] = output['aspect_words'][1]
    # output['aspect_words'] = output['aspect_words'].apply(lambda x: print(x))
    counts = output["aspect_words"].apply(lambda x: Counter(x).most_common(5))

    # print(output["aspect_words"][0][2][0])
    print(output)
    # print(counts)

    # save data to files
    output.to_csv("src/data/aspect_counts.csv")
    counts.to_csv("src/data/sentiment_counts.csv")
