import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


class Evaluator:
    """
    Evaluator Class
    """

    def __init__(self):
        self.mapping = {0: 'negative', 1: 'negative', 2: 'neutral',
                        3: 'neutral', 4: 'positive', 5: 'positive'}
        self.dataset = None
        self.train = None
        self.test = None
        self.model = None

    def summarize_review(self, data):
        data['review_polarity'] = data.groupby(
            ['reviewnumber', 'aspect'],
            as_index=False)['polarity_strength'].transform(lambda x: ','.join(x))
        data = data.drop(
            columns=['sent_idx', 'word_idx', 'word_found',
                     'polarity_strength', 'sentiment_words',
                     'intensifier_words']).drop_duplicates()

        def calculate_review_polarity(review_polarity):
            review_polarity = review_polarity.replace('[', '')
            review_polarity = review_polarity.replace(']', '')
            review_polarity = review_polarity.split(',')
            try:
                review_polarity = sum([float(polarity) for polarity in review_polarity])
            except ValueError:
                review_polarity = np.nan
            return review_polarity

        data['review_polarity'] = data['review_polarity'].apply(
            lambda x: calculate_review_polarity(x))
        data = data.dropna()
        return data

    def read_data(self, path='src/data/data_aspects_tokens.csv'):

        data = pd.read_csv(path)
        data = self.summarize_review(data)
        self.dataset = data
        x = data['review_polarity'].to_list()
        y = data['true_label'].astype(int)

        return x, y

    def sample_data(self, x, y):
        x_train = list()
        x_test = list()
        y_test = list()
        y_train = list()

        for elem_x, elem_y in zip(x, y):
            if y_train.count(self.mapping[elem_y]) < 50:
                    x_train.append([elem_x])
                    y_train.append(self.mapping[elem_y])
            elif y_test.count(self.mapping[elem_y]) < 20:
                    x_test.append([elem_x])
                    y_test.append(self.mapping[elem_y])
            except ValueError:
                pass

        print(
            'used ', str(((len(x_train) + len(x_test)) / len(x)) * 100) + '%',
            'of all reviews')

        return x_train, y_train, x_test, y_test

    def generate_train_test(self):
        x, y = self.read_data()
        x_train, y_train, x_test, y_test = self.sample_data(x, y)
        self.train = (x_train, y_train)
        self.test = (x_test, y_test)

    def train_model(self):

        model = LogisticRegression()
        model.fit(self.train[0], self.train[1])
        self.model = model

    def plot_results(self, predictions):
        fig, ax = plt.subplots()
        cm = confusion_matrix(self.test[1], predictions)
        conf = confusion_matrix(self.test[1], predictions).ravel()
        nbr_labels = len(set(self.test[1]))
        cm = conf.reshape(nbr_labels, nbr_labels)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Spectral")
        ax.set_xlabel('predicted label')
        ax.set_ylabel('true label')
        fig.savefig('test')

        fig, ax = plt.subplots()
        x = self.train[0] + self.test[0]
        y = self.train[1] + self.test[1]

        ax.scatter(y, x)
        fig.savefig('scatter')

    def evaluate(self):
        predictions = self.model.predict(self.test[0])
        accuracy = accuracy_score(self.test[1], predictions)
        print('Accuracy:', str(accuracy * 100) + '%')
        self.plot_results(predictions)
