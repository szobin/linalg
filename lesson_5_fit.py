import requests
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np


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

    x = df['time'].values
    y = df['act'].values
    z = np.polyfit(x, y, 10)
    f = np.poly1d(z)
    x_new = np.linspace(x.min(), x.max()+30, 250)
    y_new = f(x_new)

    plt.plot(x, y, '-b')
    plt.plot(x_new, y_new, '-r')
    plt.grid()
    plt.show()

    return 0


main()
