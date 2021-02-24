# AspectBasedSentimentAnalysis_ITA-Project

## Install pipenv package

Incase **pipenv** is not yet installed in the system please run:

```pip install pipenv```
(personally we would recommend including ```export PIPENV_VENV_IN_PROJECT=1``` into your .bashrc/.zshrc)

## Install pipenv

Now to get all needed dependecies run the command:

```python -m pipenv install```

## Run main

```python -m pipenv run main```

Now the code should be executing, missing packages such as Spacy models, nltk sentence tokenizer and the sentiment lexicon 
will be downloaded as part of the code. (spacy models are "de_core_news_lg", "de_core_news_sm") (the sentitment lexicon will be downloaded from 
https://raw.githubusercontent.com/sebastiansauer/pradadata/master/data-raw/germanlex.csv)

### Webscraper

The website https://www.spieletipps.de/ will be scraped for the games we indicated, extracting reviews and
the stars given for aspects on the website.

### Preprossesor

Each review will be normalized, and tokenized based on the parameters given in the main.

### Annotator

For each review we look up words that match any of the strings we consider being part of one of the aspects
and save the sentece index and the word index.

### Sentiment Detector

For each found aspect we use spacy to tag and create a dependecy tree of the sentece. We use this dependecy tree to 
find adjectives describing the aspect and look up their polarity in a sentiment lexicon.


## Data

The data will be saved in ```src/data```

the file are namely:
- ```data_raw.csv```
- ```data_preprocessed.csv```
- ```data_aspect_tokens.csv```


