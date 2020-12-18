import os
import json
import re
import pprint


def read_dataset(path):
    files = os.listdir(path)
    dataset = list()
    for data in files:
        with open(path + "/" + data, "r") as doc:
            data = json.load(doc)
            dataset.append(data)
    return dataset


def normalize_review(text):

    text = text.replace("\n", "")
    text = text.strip("\n")
    text = text.lower()

    return text


def split_sentences(text):

    sentences = re.split("\.|\?|!", text)
    return sentences


if __name__ == "__main__":

    dataset = read_dataset("./data/raw")

    preprocessed = list()

    aspects = ["grafik", "sound", "steuerung", "atmosphäre"]

    for game in dataset:

        for review in game["reviews"]:

            if review["rating"] != {}:

                normalized_review = normalize_review(review["text"])
                sentences = split_sentences(normalized_review)

                for sentence in sentences:
                    opinions = []

                    p = re.compile("(grafik\w*|optik\w*)")
                    occurences_g = re.findall(p, sentence)
                    if len(occurences_g) > 0:
                        for occurence in occurences_g:
                            sentence = re.sub(p, "GRAPHICS", sentence)
                            graphics_dict = {
                                "sentiment": review["rating"]["Grafik"],
                                "aspect": "GRAPHICS",
                                "original-entity": occurence,
                            }
                            opinions.append(graphics_dict)

                    # check "sound"
                    p = re.compile("sound\w*|klang\w*|ton\w*|akustik\w*")
                    occurences_s = re.findall(p, sentence)

                    if len(occurences_s) > 0:
                        for occurence in occurences_s:
                            sentence = re.sub(p, "SOUND", sentence)
                            sound_dict = {
                                "sentiment": review["rating"]["Sound"],
                                "aspect": "SOUND",
                                "original-entity": occurence,
                            }
                            opinions.append(sound_dict)

                    # check "steuerung"
                    p = re.compile("steuerung\w*|bedienung\w*")
                    occurences_c = re.findall(p, sentence)

                    if len(occurences_c) > 0:
                        for occurence in occurences_c:
                            sentence = re.sub(p, "CONTROLS", sentence)
                            controls_dict = {
                                "sentiment": review["rating"]["Steuerung"],
                                "aspect": "CONTROLS",
                                "original-entity": occurence,
                            }
                            opinions.append(controls_dict)

                    # check "atmosphäre"
                    p = re.compile("atmosphäre\w*|stimmung\w*")
                    occurences_a = re.findall(p, sentence)

                    if len(occurences_a) > 0:
                        for occurence in occurences_a:
                            sentence = re.sub(p, "ATMOSPHERE", sentence)
                            atmosphere_dict = {
                                "sentiment": review["rating"]["Atmosph\u00e4re"],
                                "aspect": "ATMOSPHERE",
                                "original-entity": occurence,
                            }
                            opinions.append(atmosphere_dict)

                    if opinions != []:
                        sentence_dict = {
                            "opinions": opinions,
                            "text": sentence,
                        }  # add id of review
                        preprocessed.append(sentence_dict)

    out = json.dumps(preprocessed)
    pprint.pprint(preprocessed)
    print(len(preprocessed), "sentences")
    with open("data/preprocessed/data.json", "w") as doc:
        doc.write(out)
