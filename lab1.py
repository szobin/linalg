from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from pandas import read_csv
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({'font.size': 12})
np.warnings.filterwarnings('ignore')


def main():

    # load & prepare
    df = read_csv('NAICExpense.csv', ',')
    df = df[['EXPENSES', 'RBC', 'STAFFWAGE', 'AGENTWAGE', 'LONGLOSS', 'SHORTLOSS']]
    means = df.mean()
    df = df.fillna(means)
    print(df.info())

    # correlations
    x = df[['RBC', 'STAFFWAGE', 'AGENTWAGE', 'LONGLOSS', 'SHORTLOSS']]
    y = df['EXPENSES']

    corr = df.corr()
    print(corr)
    r = sns.heatmap(corr, linewidths=.5, annot=True, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
    fig = r.get_figure()
    fig.savefig('fig1_1.png')

    # linear regression
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    # ridge regression with low alpha
    rr = Ridge(alpha=0.01)
    rr.fit(x_train, y_train)

    # ridge regression with high alpha
    rr100 = Ridge(alpha=100)
    rr100.fit(x_train, y_train)

    train_score = lr.score(x_train, y_train)
    test_score = lr.score(x_test, y_test)

    ridge_train_score = rr.score(x_train, y_train)
    ridge_test_score = rr.score(x_test, y_test)

    ridge_train_score100 = rr100.score(x_train, y_train)
    ridge_test_score100 = rr100.score(x_test, y_test)

    print("linear regression train score:", train_score)
    print("linear regression test score:", test_score)

    print("ridge regression train score low alpha:", ridge_train_score)
    print("ridge regression test score low alpha:", ridge_test_score)
    print("ridge regression train score high alpha:", ridge_train_score100)
    print("ridge regression test score high alpha:", ridge_test_score100)

    plt.close()
    plt.plot(rr.coef_, alpha=0.7, linestyle='none', marker='*', markersize=5, color='red', label=r'Ridge; $\alpha = 0.01$', zorder=7)  # zorder for ordering the markers
    plt.plot(rr100.coef_, alpha=0.5, linestyle='none', marker='d', markersize=6, color='blue', label=r'Ridge; $\alpha = 100$')  # alpha here is for transparency
    plt.plot(lr.coef_, alpha=0.4, linestyle='none', marker='o', markersize=7, color='green', label='Linear Regression')
    plt.xlabel('Coefficient Index', fontsize=16)
    plt.ylabel('Coefficient Magnitude', fontsize=16)
    plt.legend(fontsize=13, loc=4)
    plt.savefig('fig1_2.png')


if __name__ == "__main__":
    main()
