import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


class Evaluator:

    def __init__(self):
        self.mapping = {0: 'negative', 1: 'negative', 2: 'neutral',
                        3: 'neutral', 4: 'positive', 5: 'positive'}
        self.dataset = None
        self.train = None
        self.test = None
        self.model = None

    def read_data(self, path='src/data/data_aspects_tokens.csv'):

        data = pd.read_csv(path)
        x = data['polarity_strength'].to_list()
        y = data['true_label'].astype(int)

        return x, y

    def sample_data(self, x, y):
        x_train = list()
        x_test = list()
        y_test = list()
        y_train = list()

        for elem_x, elem_y in zip(x, y):
            elem_x = elem_x.replace(']', '')
            elem_x = elem_x.replace('[', '')
            try:
                elem_x = float(elem_x)
                if y_train.count(self.mapping[elem_y]) < 400:
                    x_train.append([elem_x])
                    y_train.append(self.mapping[elem_y])
                elif y_test.count(self.mapping[elem_y]) < 100:
                    x_test.append([elem_x])
                    y_test.append(self.mapping[elem_y])
            except:
                pass

        print(
            'used ', str(((len(x_train) + len(x_test)) / len(x)) * 100) + '%',
            'of all sentences')

        return x_train, y_train, x_test, y_test

    def generate_train_test_from_path(self):
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
        cm = conf.reshape(3, 3)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Spectral")
        ax.set_xlabel('predicted label')
        ax.set_ylabel('true label')
        fig.savefig('test')

    def evaluate(self):
        predictions = self.model.predict(self.test[0])
        accuracy = accuracy_score(self.test[1], predictions)
        print('Accuracy:', str(accuracy * 100) + '%')
        self.plot_results(predictions)


evaluator = Evaluator()
evaluator.generate_train_test_from_path()
evaluator.train_model()
evaluator.evaluate()
