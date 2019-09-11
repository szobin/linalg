import math
import pandas as pd
import numpy as np
from functools import reduce


def avg(dd, p=2):
    nn = len(dd)
    s = sum(dd)
    return round(s / nn, p)


def sum2(ss, d):
    return ss+d*d


def std(dd, p=2):
    nn = len(dd)
    s = sum(dd)

    s21 = reduce(sum2, dd, 0)
    s22 = reduce(lambda ss, d: ss+d*d, dd, 0)
    s2 = sum([d*d for d in dd])

    if (s21 != s2) or (s22 != s2):
        raise Exception("error reduce")

    v = (s2 - s * s / nn) / nn
    disp = v * nn / (nn - 1)
    return round(math.sqrt(disp), p)


def corr(dd1, dd2, p=2):
    nn = len(dd1)
    sum_d1 = sum(dd1)
    sum_d2 = sum(dd2)
    sum_2d1 = sum([d * d for d in dd1])
    sum_2d2 = sum([d * d for d in dd2])
    sum_d1_d2 = sum([d1 * d2 for d1, d2 in zip(dd1, dd2)])

    a = nn * sum_d1_d2 - sum_d1 * sum_d2
    b = nn*sum_2d1 - sum_d1*sum_d1
    c = nn*sum_2d2 - sum_d2*sum_d2

    v_corr = a / math.sqrt(b) / math.sqrt(c)
    return round(v_corr, p)


def sums(s, d):
    return s[0]+1, s[1]+d[0], s[2]+d[1], s[3]+d[0]*d[0], s[4]+d[1]*d[1], s[5]+d[0]*d[1]


def corr_reduce(dd1, dd2, p=2):
    (nn, sum_d1, sum_d2, sum_2d1, sum_2d2, sum_d1_d2) = reduce(sums, zip(dd1, dd2), [0] * 6)
    a = nn * sum_d1_d2 - sum_d1 * sum_d2
    b = nn*sum_2d1 - sum_d1*sum_d1
    c = nn*sum_2d2 - sum_d2*sum_d2

    v_corr = a / math.sqrt(b) / math.sqrt(c)
    return round(v_corr, p)


def get_vals(d):
    return [d[0]*d[0], d[1]*d[1], d[0]*d[1]]


def corr_map(dd1, dd2, p=2):
    dd = np.array(list(map(get_vals, zip(dd1, dd2))))

    nn = len(dd1)
    sum_d1 = sum(dd1)
    sum_d2 = sum(dd2)
    sum_2d1 = sum(dd[:, 0])
    sum_2d2 = sum(dd[:, 1])
    sum_d1_d2 = sum(dd[:, 2])

    a = nn * sum_d1_d2 - sum_d1 * sum_d2
    b = nn*sum_2d1 - sum_d1*sum_d1
    c = nn*sum_2d2 - sum_d2*sum_d2

    v_corr = a / math.sqrt(b) / math.sqrt(c)
    return round(v_corr, p)


c1 = [1.2, 1.7, 3.5, 2.1, 3.1, 5.8, 1.9, 3.2, 4.6, 2.7]
c2 = [2.1, 4.3, 2.6, 4.7, 1.1, 5.2, 6.1, 3.7, 4.0, 1.9]

print('d1 avg=', avg(c1))
print('d2 avg=', avg(c2))

print('d1 std=', std(c1))
print('d2 std=', std(c2))

print('d1/d2 corr=', corr(c1, c2, 4))
print('d1/d2 corr_reduce=', corr_reduce(c1, c2, 4))
print('d1/d2 corr_map=', corr_map(c1, c2, 4))

df = pd.DataFrame()
df["c1"] = np.array(c1)
df["c2"] = np.array(c2)
df_avg = df.mean()
print(df_avg)

df_std = df.std()
print(df_std)

df_corr = df.corr()
print(df_corr)
