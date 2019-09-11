import requests
import pandas as pd
import scipy
import scipy.fftpack
import io
# import matplotlib.pyplot as plt


def main():
    r = requests.get("http://www.sidc.be/silso/INFO/snmtotcsv.php")
    if r.status_code != 200:
        return -1

    csv = "year;month;time;act;a1;a2;a3\n"+r.content.decode('utf-8')
    df = pd.read_csv(io.StringIO(csv), delimiter=";")
    df_1 = df[["time", "act"]]  # .tail(12*12)
    p = df_1.plot(x="time")
    fig = p.get_figure()
    fig.savefig('fig_l_2.png')

    return 0


main()
