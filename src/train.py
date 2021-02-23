import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('src/data/data_aspects_tokens.csv')
x = data['polarity_strength'].to_list()
y = data['true_label'].astype(int)

prep_x = list()
prep_y = list()

for elem_x, elem_y in zip(x, y):
    elem_x = elem_x.replace(']', '')
    elem_x = elem_x.replace('[', '')
    try:
        elem_x = float(elem_x)
        if prep_y.count(elem_y) < 100:
            prep_x.append([elem_x])
            prep_y.append(elem_y)
    except:
        pass

x_train, x_test, y_train, y_test = train_test_split(
    prep_x, prep_y, test_size=0.25, random_state=42)
