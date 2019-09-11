from pandas import read_csv
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    x = np.array([[-3, 7], [1, 5], [1, 2], [-2, 0], [2, 3], [-4, 0],
                  [-1, 1], [1, 1], [-2, 2], [2, 7], [-4, 1], [-2, 7]])
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


def find_2_max_regresors(model, X_train, X_test, y_train, y_test):
    max_score = -1
    max_c1 = None
    max_c2 = None
    for c1 in X_train.columns:
        for c2 in X_train.columns:
            if c1 == c2:
                continue

            new_x_train = X_train[[c1, c2]]
            new_x_test = X_test[[c1, c2]]

            model.fit(new_x_train, y_train)

            y_test_pred = model.predict(new_x_test)
            score = metrics.accuracy_score(y_test, y_test_pred)
            if score > max_score:
                max_c1 = c1
                max_c2 = c2
                max_score = score

    return max_c1, max_c2


def draw_classification_area(model, c1, c2, X_train, X_test, y_train, y_test, title):
    x_train_draw = scale(X_train[[c1, c2]].values)
    x_test_draw = scale(X_test[[c1, c2]].values)

    model.fit(x_train_draw, y_train)

    x_min, x_max = x_train_draw[:, 0].min() - 1, x_train_draw[:, 0].max() + 1
    y_min, y_max = x_train_draw[:, 1].min() - 1, x_train_draw[:, 1].max() + 1

    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
    pred = pred.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.figure()
    plt.pcolormesh(xx, yy, pred, cmap=cmap_light)
    plt.scatter(x_train_draw[:, 0], x_train_draw[:, 1],
                c=y_train, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.title("%s %s:%s score: %.0f percents" % (title, c1, c2, model.score(x_test_draw, y_test) * 100))
    plt.savefig('fig3_{}.png'.format(title))


def decision_tree_model(x, Y):
    print('Decision tree model')
    X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.20, random_state=0)

    tuned_parameters = {'criterion': ['gini', 'entropy'],
                        'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150]}
    model_dt = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=5)

    model_dt.fit(X_train, y_train)
    print('best model parameters')
    print(model_dt.best_params_)

    y_train_pred = model_dt.predict(X_train)
    y_test_pred = model_dt.predict(X_test)
    acc_train = metrics.accuracy_score(y_train, y_train_pred)
    acc_test = metrics.accuracy_score(y_test, y_test_pred)

    print()
    print("DecisionTree classifier")
    print('Accuracy on train set: {:.2f}'.format(acc_train))
    print('Accuracy on test set: {:.2f}'.format(acc_test))

    model_dt = DecisionTreeClassifier(**model_dt.best_params_)
    c1, c2 = find_2_max_regresors(model_dt, X_train, X_test, y_train, y_test)
    draw_classification_area(model_dt, c1, c2, X_train, X_test, y_train, y_test, 'decision_tree')


def boosting_model(x, Y):
    print()
    print('Boosting model')
    X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.20, random_state=0)
    tuned_parameters = {
        "loss": ["deviance"],
        "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
        # "min_samples_split": np.linspace(0.1, 0.5, 12),
        # "min_samples_leaf": np.linspace(0.1, 0.5, 12),
        "max_depth": [3, 5, 8],
        "max_features": ["log2", "sqrt"],
        "criterion": ["friedman_mse", "mae"],
        # "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
        "n_estimators": [10]
    }
    model_gb = GridSearchCV(GradientBoostingClassifier(), tuned_parameters, cv=5)

    model_gb.fit(X_train, y_train)
    print('best model parameters')
    print(model_gb.best_params_)

    y_train_pred = model_gb.predict(X_train)
    y_test_pred = model_gb.predict(X_test)

    acc_train = metrics.accuracy_score(y_train, y_train_pred)
    acc_test = metrics.accuracy_score(y_test, y_test_pred)

    print()
    print("Boosting classifier")
    print('Accuracy on train set: {:.2f}'.format(acc_train))
    print('Accuracy on test set: {:.2f}'.format(acc_test))

    model_gb = GradientBoostingClassifier(**model_gb.best_params_)
    c1, c2 = find_2_max_regresors(model_gb, X_train, X_test, y_train, y_test)
    draw_classification_area(model_gb, c1, c2, X_train, X_test, y_train, y_test, 'boosting')


def random_forest_model(x, Y):
    print()
    print('Random forest model')
    X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.20, random_state=0)

    tuned_parameters = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy']
    }
    model_rf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5)

    model_rf.fit(X_train, y_train)
    print('best model parameters')
    print(model_rf.best_params_)

    y_train_pred = model_rf.predict(X_train)
    y_test_pred = model_rf.predict(X_test)

    acc_train = metrics.accuracy_score(y_train, y_train_pred)
    acc_test = metrics.accuracy_score(y_test, y_test_pred)

    print()
    print("Random Forest classifier")
    print('Accuracy on train set: {:.2f}'.format(acc_train))
    print('Accuracy on test set: {:.2f}'.format(acc_test))

    model_rf = RandomForestClassifier(**model_rf.best_params_)
    c1, c2 = find_2_max_regresors(model_rf, X_train, X_test, y_train, y_test)
    draw_classification_area(model_rf, c1, c2, X_train, X_test, y_train, y_test, 'random_forest')


def main():
    features, labels = prepare_data("crx.data.csv")
    decision_tree_model(features, labels)
    boosting_model(features, labels)
    random_forest_model(features, labels)


if __name__ == "__main__":
    main()
