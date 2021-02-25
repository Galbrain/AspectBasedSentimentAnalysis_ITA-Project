import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB

mapping = {0: 'negative', 1: 'negative', 2: 'neutral',
           3: 'neutral', 4: 'positive', 5: 'positive'}


def read_data(path='src/data/data_aspects_tokens.csv'):

    data = pd.read_csv(path)
    x = data['polarity_strength'].to_list()
    y = data['true_label'].astype(int)

    return x, y


def sample_data(x, y):
    x_train = list()
    x_test = list()
    y_test = list()
    y_train = list()

    for elem_x, elem_y in zip(x, y):
        elem_x = elem_x.replace(']', '')
        elem_x = elem_x.replace('[', '')
        try:
            elem_x = float(elem_x)
            if y_train.count(mapping[elem_y]) < 400:
                x_train.append([elem_x])
                y_train.append(mapping[elem_y])
            elif y_test.count(mapping[elem_y]) < 100:
                x_test.append([elem_x])
                y_test.append(mapping[elem_y])
        except:
            pass

    print(
        'used ', str(((len(x_train) + len(x_test)) / len(x)) * 100) + '%',
        'of all sentences')

    return x_train, y_train, x_test, y_test


def train_model(x_train, y_train):

    naive_bayes = GaussianNB()
    naive_bayes.fit(x_train, y_train)
    lg = LogisticRegression()
    lg.fit(x_train, y_train)

    return lg


def predict(model, x_test):
    y_pred = model.predict(x_test)
    return y_pred


def evaluate(model, y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    conf = confusion_matrix(y_test, y_pred).ravel()
    cm = conf.reshape(3, 3)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Spectral")
    ax.set_xlabel('predicted label')
    ax.set_ylabel('true label')
    fig.savefig('test')


x, y = read_data()
x_train, y_train, x_test, y_test = sample_data(x, y)
model = train_model(x_train, y_train)
prediction = predict(model, x_test)
evaluate(model, prediction, y_test)
