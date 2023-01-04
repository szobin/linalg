import pandas as pd
import numpy as np
from intervals import Segment, SegType


def main():
    s = Segment(SegType.CLOSE, -5, 5)

    df = pd.DataFrame(np.linspace(s.start, s.finish, num=100), columns=["TIME"])
    x = df["TIME"]

    pp = [
          (1, 10, -10, -15),
          (2, 8, -5, -3),
          (3, -2, 5, 16),
          ]

    for p in pp:
        df['P'+str(p[0])] = p[1] * x**2 + p[2] * x + p[3]

    plot = df.plot(x="TIME", grid=True, title="Parabolas")
    fig = plot.get_figure()
    fig.savefig("graph.png")


if __name__ == "__main__":
    main()
