import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA as sklearnPCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn import linear_model

filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
filled_colors = ('red', 'green', 'blue')

data_cols_cont_descr = {
    "a1": False,
    "a2": True,
    "a3": True,
    "a4": False,
    "a5": False,
    "a6": False,
    "a7": False,
    "a8": True,
    "a9": False,
    "a10": False,
    "a11": True,
    "a12": False,
    "a13": False,
    "a14": True,
    "a15": True,
    "a16": False,
}


# prepare data
def prepare_test_data():
    x = np.array([[-3, 7], [1, 5], [1, 2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
    Y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])
    return x, Y


def prepare_data(fn):
    if fn is None:
        return prepare_test_data()

    le = preprocessing.LabelEncoder()

    df = read_csv(fn, ';')

    y = le.fit_transform(df["a16"])

    x = df.drop('a16', 1)

    for c in x:
        dc = x[c]
        if data_cols_cont_descr[c]:
            dc = pd.to_numeric(dc, errors='coerce')
            # print(dc)
            dfc = dc.to_frame(c)
            m = dfc.mean()
            dfc = dfc.fillna(m)
            x[c] = dfc[c]
        else:
            x[c] = le.fit_transform(dc)

    print("Data control")
    print(x.info())

    return x, y


# Naive bayes
def bayes_model(x, Y):
    X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.20, random_state=0)

    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)

    y_train_pred = model_nb.predict(X_train)
    y_test_pred = model_nb.predict(X_test)
    acc_train = metrics.accuracy_score(y_train, y_train_pred)
    acc_test = metrics.accuracy_score(y_test, y_test_pred)

    print()
    print("Naive Bayes classifier")
    print('Accuracy on train set: {:.2f}'.format(acc_train))
    print('Accuracy on test set: {:.2f}'.format(acc_test))


# k-nn == K nearest neighbour
def knn_model(x, Y):
    X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.20, random_state=0)

    print()
    print("K-NN classifier")
    for n_neighbors in range(5, 10):
        print("n-neighbors = {}".format(n_neighbors))

        model_knn = KNeighborsClassifier(n_neighbors)
        model_knn.fit(X_train, y_train)
        print('Accuracy on training set: {:.2f}'
              .format(model_knn.score(X_train, y_train)))
        print('Accuracy on test set: {:.2f}'
              .format(model_knn.score(X_test, y_test)))


# logistic regression
def logistic_model(x, Y):
    X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.20, random_state=0)

    model_clf = linear_model.LogisticRegression(solver='lbfgs', max_iter=5000)

    model_clf.fit(X_train, y_train)

    print()
    print("Logistic regression classifier (for lbfgs solver)")
    print('Accuracy on training set: {:.2f}'
          .format(model_clf.score(X_train, y_train)))
    print('Accuracy on test set: {:.2f}'
          .format(model_clf.score(X_test, y_test)))


def factor_view(x, Y, factor):
    yy = x[factor]
    xx = x.drop(factor, 1)
    xx['y'] = Y

    X_norm = (xx - xx.min()) / (xx.max() - xx.min())
    factor_v = sorted(yy.unique())
    y_v = sorted(X_norm['y'].unique())

    # lda = LDA(n_components=2)  # 2-dimensional LDA
    # transformed = pd.DataFrame(lda.fit_transform(X_norm, Y))

    pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
    transformed = pd.DataFrame(pca.fit_transform(X_norm))

    fig = plt.figure()
    ax = plt.subplot(111)

    for i, c in enumerate(factor_v):
        y0 = transformed[(X_norm['y'] == y_v[0]) & (yy == c)]
        y1 = transformed[(X_norm['y'] == y_v[1]) & (yy == c)]

        plt.scatter(y0[0], y0[1], label='{}={} app'.format(factor, c), c=filled_colors[i], marker=filled_markers[0])
        plt.scatter(y1[0], y1[1], label='{}={} ref'.format(factor, c), c=filled_colors[i], marker=filled_markers[1])

    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    plt.legend(loc='upper center',  bbox_to_anchor=(1.45, 0.8), ncol=1)
    plt.savefig('fig2_factor.png')


def main():
    features, labels = prepare_data("crx.data.csv")
    bayes_model(features, labels)
    knn_model(features, labels)
    logistic_model(features, labels)
    factor_view(features, labels, 'a13')


if __name__ == "__main__":
    main()
