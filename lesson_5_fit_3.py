import requests
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression


def get_silso():
    r = requests.get("http://www.sidc.be/silso/INFO/snmtotcsv.php")
    if r.status_code != 200:
        return None

    csv = "year;month;time;act;a1;a2;a3\n"+r.content.decode('utf-8')
    df_csv = pd.read_csv(io.StringIO(csv), delimiter=";")
    return df_csv[["time", "act"]]


def main():
    df = get_silso()
    if df is None:
        return -1

    x = np.array([(t, a) for t, a in zip(df['time'].values, df['act'].values)])
    y = np.array(range(x.size))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)

    model_lr = LinearRegression()
    model_rr = Ridge()
    model_lgr = LogisticRegression()

    model_lr.fit(x_train, y_train)
    model_rr.fit(x_train, y_train)
    model_lgr.fit(x_train, y_train)

    train_score = model_lr.score(x_train, y_train)
    test_score = model_lr.score(x_test, y_test)
    print("Linear regression train score:", train_score)
    print("Linear regression test score:", test_score)

    train_score = model_rr.score(x_train, y_train)
    test_score = model_rr.score(x_test, y_test)
    print("Ridge regression train score:", train_score)
    print("Ridge regression test score:", test_score)

    plt.plot(x, y, '-b')
    plt.grid()
    plt.show()

    return 0


main()
