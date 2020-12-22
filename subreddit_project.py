from time import process_time
import praw
import numpy as np
import pandas as pd
import csv


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from warnings import filterwarnings
import matplotlib.pyplot as plt
import re

#####################

data = pd.read_csv('lib_conserv30.csv')


def cleaned(text):
    if (text == "[deleted]") or (text == "[removed]"):
        text = ""
    text = re.sub(r"\n", "", text)
    text = text.lower()
    text = re.sub(r"\d", "", text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    return text


data['text'] = data['text'].apply(str)
data['cleaned'] = data['text'].apply(lambda x: cleaned(x))


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=70)

X = data['cleaned'].to_numpy()
y = data['target'].to_numpy()

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


reddit_pipeline = Pipeline(
    [('CVec', CountVectorizer(stop_words='english')), ('Tfidf', TfidfTransformer())])

X_train_tranformed = reddit_pipeline.fit_transform(X_train)
X_test_tranformed = reddit_pipeline.transform(X_test)

models = {
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
    'Multinomial NB': MultinomialNB(),
    "Neural Network (MLP)": MLPClassifier(max_iter=300),
}

no_models = len(models.keys())


def batch_classify(X_train_tranformed, y_train, X_test_tranformed, y_test, verbose=True):
    df_results = pd.DataFrame(data=np.zeros(shape=(no_models, 4)), columns=[
                              'Model', 'Accuracy', 'AUC', 'Training time'])
    count = 0
    for key, model in models.items():
        t_start = process_time()
        model.fit(X_train_tranformed, y_train)
        t_stop = process_time()
        t_elapsed = t_stop - t_start
        y_predicted = model.predict(X_test_tranformed)

        auc_ = roc_auc_score(y_test, y_predicted)

        df_results.loc[count, 'Model'] = key
        df_results.loc[count, 'Accuracy'] = model.score(
            X_test_tranformed, y_test)
        df_results.loc[count, 'AUC'] = auc_
        df_results.loc[count, 'Training time'] = t_elapsed
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=key, f=t_elapsed))

        probs = model.predict_proba(X_test_tranformed)
        probs = probs[:, 1]

        # calculate roc curve
        fpr, tpr, thresholds = roc_curve(y_test, probs, pos_label=1)

        plt.plot([0, 1], [0, 1], linestyle='--', color='#003366')
        plt.plot(fpr, tpr, color='orange',
                 label='ROC Curve (AUC = %0.2f)' % auc_)

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(str(key) + ' ROC')
        plt.legend(loc="lower right")
        # show the plot
        plt.show()
        plt.savefig('graph' + str(key) + '.png')

        count += 1

    return df_results


df_results = batch_classify(
    X_train_tranformed, y_train, X_test_tranformed, y_test)
print(df_results.sort_values(by='Accuracy', ascending=False))
