import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

mapping = {0: 'negative', 1: 'negative', 2: 'neutral',
           3: 'neutral', 4: 'positive', 5: 'positive'}

data = pd.read_csv('src/data/data_aspects_tokens.csv')
x = data['polarity_strength'].to_list()
y = data['true_label'].astype(int)

prep_x = list()
prep_y = list()

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

x_train, x_test, y_train, y_test = train_test_split(
    prep_x, prep_y, test_size=0.25, random_state=42)

print('used ', str(((len(x_train)+len(x_test))/len(x))*100) + '%', 'of all sentences')

naive_bayes = GaussianNB()
naive_bayes.fit(x_train, y_train)
lg = LogisticRegression()
lg.fit(x_train, y_train)

y_pred = naive_bayes.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
fig, ax = plt.subplots()
cm = confusion_matrix(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred).ravel()
cm = conf.reshape(3, 3)
sns.heatmap(cm, annot=True, fmt="d", cmap="Spectral")
ax.set_xlabel('predicted label')
ax.set_ylabel('true label')
fig.savefig('test_nb')

y_pred = lg.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
fig, ax = plt.subplots()
cm = confusion_matrix(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred).ravel()
cm = conf.reshape(3, 3)
sns.heatmap(cm, annot=True, fmt="d", cmap="Spectral")
ax.set_xlabel('predicted label')
ax.set_ylabel('true label')
fig.savefig('test_lg')
