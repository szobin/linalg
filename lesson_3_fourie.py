import requests
import pandas as pd
import numpy as np
import scipy.fftpack as fft
import io
import matplotlib.pyplot as plt


def fft_sample():
    f = 10  # Frequency, in cycles per second, or Hertz
    f_s = 100  # Sampling rate, or number of measurements per second

    t = np.linspace(0, 2, 2 * f_s, endpoint=False)
    x = np.sin(f * 2 * np.pi * t)
    '''
    fig, ax = plt.subplots()
    ax.plot(t, x)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Signal amplitude')
    plt.show()
    '''

    dl = len(x)
    dl2 = dl // 2

    fm = fft.fft(x)
    freqs = fft.fftfreq(dl, 1/f_s)

    fig, ax = plt.subplots()

    # ax.stem(freqs, np.abs(fm), use_line_collection=True)
    ax.stem(freqs[:dl2], np.abs(fm)[:dl2]/f_s, use_line_collection=True)

    ax.set_xlabel('Frequency in Hertz [Hz]')
    ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
    # ax.set_xlim(-f_s / 2, f_s / 2)
    # ax.set_ylim(-5, 110)
    plt.show()


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
    sa = sum(df["act"])

    yf = fft.fft(df['act'].values)
    print(len(yf))
    # print(yf)
    xf = fft.fftfreq(yf.size, 1/12)

    xf1 = 1/xf[3:100]
    yf1 = np.abs(yf[3:100])/sa

    plt.stem(xf1, yf1, use_line_collection=True)

    m = max(yf1)
    p_max = np.array(list(filter(lambda p: p[1] > 0.8*m, zip(xf1, yf1))))
    # print(p_max)
    plt.scatter(p_max[:, 0], p_max[:, 1], s=150, color="r")

    # plt.plot(xf, yf)
    plt.xlim((0, 20))
    plt.grid()
    plt.show()
    return 0


# fft_sample()
main()
