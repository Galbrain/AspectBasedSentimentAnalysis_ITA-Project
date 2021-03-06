from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


class Evaluator:
    """
    Evaluator Class
    """

    def __init__(self):
        self.mapping = {
            0: "negative",
            1: "negative",
            2: "neutral",
            3: "neutral",
            4: "positive",
            5: "positive",
        }
        self.dataset = None
        self.train = None
        self.test = None
        self.model = None

    def summarize_review(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        calculates overall sentiment for each aspect of a single review
        using the polarity strength of each occurence of the aspect

        Args:
            data (pd.DataFrame): Dataframe containing information about
            the aspects, their polarities and their respective review

        Returns:
            pd.DataFrame: Dataframe with overall polarity for each aspect in a review
        """

        data["review_polarity"] = data.groupby(
            ["reviewnumber", "aspect"], as_index=False
        )["polarity_strength"].transform(lambda x: ",".join(x))
        data = data.drop(
            columns=[
                "sent_idx",
                "word_idx",
                "word_found",
                "polarity_strength",
                "sentiment_words",
                "intensifier_words",
            ]
        ).drop_duplicates()

        def calculate_review_polarity(review_polarity):
            review_polarity = review_polarity.replace("[", "")
            review_polarity = review_polarity.replace("]", "")
            review_polarity = review_polarity.split(",")
            try:
                review_polarity = np.mean(
                    [float(polarity) for polarity in review_polarity]
                )
            except ValueError:
                review_polarity = np.nan
            return review_polarity

        data["review_polarity"] = data["review_polarity"].apply(
            lambda x: calculate_review_polarity(x)
        )
        data = data.dropna()
        return data

    def read_data(
        self, path: str = "src/data/data_aspects_tokens.csv"
    ) -> Tuple[list, list]:
        """
        reads csv data from specified path

        Args:
            path (str, optional): Path to the csv file. Defaults to 'src/data/data_aspects_tokens.csv'.

        Returns:
            Tuple[list, list]: list with review polarity for an aspect and list with corresponding true label
        """
        data = pd.read_csv(path)
        data = self.summarize_review(data)
        self.dataset = data
        x = data["review_polarity"].to_list()
        y = data["true_label"].astype(int)

        return x, y

    def sample_data(self, x: list, y: list) -> Tuple[list, list, list, list]:
        """
        samples data so that possible labels are equally distributed

        Args:
            x (list): x-data (review polarity for aspect)
            y (list): y-data (true label for aspect in review)

        Returns:
            Tuple[list, list, list, list]: dataset splitted into train and test set
        """
        y = [self.mapping[i] for i in y]

        x_train = list()
        x_test = list()
        y_test = list()
        y_train = list()

        min_label = min([y.count(i) for i in list(set(y))])
        nbr_labels = len(set(y))
        data_size = min_label * len(set(y))

        train_size = int(0.66 * data_size)
        test_size = data_size - train_size

        for elem_x, elem_y in zip(x, y):
            if y_train.count(elem_y) < int(train_size / nbr_labels):
                x_train.append([elem_x])
                y_train.append(elem_y)

            elif y_test.count(elem_y) < int(test_size / nbr_labels):
                x_test.append([elem_x])
                y_test.append(elem_y)

        return x_train, y_train, x_test, y_test

    def generate_train_test(self):
        """
        generates train and test set
        """
        x, y = self.read_data()
        x_train, y_train, x_test, y_test = self.sample_data(x, y)
        self.train = (x_train, y_train)
        self.test = (x_test, y_test)

    def train_model(self):
        """
        trains model
        """
        model = LogisticRegression()
        model.fit(self.train[0], self.train[1])
        self.model = model

    def plot_results(self, predictions: list):
        """
        creates and saves a confusion matrix and boxplot

        Args:
            predictions (list): list containing the predicted labels for the test data
        """
        fig, ax = plt.subplots()
        cm = confusion_matrix(self.test[1], predictions)
        conf = confusion_matrix(self.test[1], predictions).ravel()
        nbr_labels = len(set(self.test[1]))
        cm = conf.reshape(nbr_labels, nbr_labels)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Spectral")
        ax.set_xlabel("predicted label")
        ax.set_ylabel("true label")
        fig.savefig("confusion_matrix")

        fig, ax = plt.subplots()
        x = self.train[0] + self.test[0]
        y = self.train[1] + self.test[1]
        x = [i[0] for i in x]
        y = [i for i in y]
        results = pd.DataFrame({"polarity strength": x, "true label": y})
        sns.boxplot(data=results, x="true label", y="polarity strength")
        fig.savefig("boxplot")

    def evaluate(self):
        """
        performs evaluation of the results including visualization
        """
        predictions = self.model.predict(self.test[0])
        accuracy = accuracy_score(self.test[1], predictions)
        print("Accuracy:", str(accuracy * 100) + "%")
        self.plot_results(predictions)
