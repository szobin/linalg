import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import io


def get_silso():
    r = requests.get("http://www.sidc.be/silso/INFO/snmtotcsv.php")
    if r.status_code != 200:
        return -1

    csv = "year;month;time;act;a1;a2;a3\n" + r.content.decode('utf-8')
    df = pd.read_csv(io.StringIO(csv), delimiter=";")
    df_1 = df[["time", "act"]]  # .tail(12*12)
    return df_1


def fit(x, a, b, c, d):
    return a*np.sin(b*x + c) + d


df = get_silso()
signal = np.array(df["act"].values)
xs = np.array(df["time"].values)

# guess = (signal.ptp()/2, 10, 0, signal.mean())
fitting_parameters, covariance = optimize.curve_fit(fit, xs, signal)  # , p0=guess
a, b, c, d = fitting_parameters
print(a, b, c, d)

fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(xs, signal)

new_xs = np.linspace(xs.min(), xs.max(), len(signal)*1.5)
new_ys = fit(new_xs, a, b, c, d)
ax[1].plot(new_xs, new_ys)

plt.xlim(xs.min(), xs.max())
plt.show()
