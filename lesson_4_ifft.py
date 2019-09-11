import requests
import pandas as pd
import numpy as np
import scipy.fftpack as fft
import io
import matplotlib.pyplot as plt


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

    signal = df['act'].values
    t = df['time'].values

    yf = fft.rfft(signal)
    yf_cut = yf.copy()
    # freq = fft.fftfreq(yf.size, d=t[1]-t[0])
    p = abs(yf_cut).max() / 20
    yf_cut[abs(yf_cut) < p] = 0

    new_signal = fft.irfft(yf_cut)

    plt.plot(t, signal, '-b')
    plt.plot(t, new_signal, '-r')
    plt.grid()
    plt.show()
    return 0


main()
