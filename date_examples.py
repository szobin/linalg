from datetime import datetime, timedelta
import zlib, base64


def get_minerva_year_seconds():
    today = datetime.now()
    minerva = datetime(2020, 4, 29)
    dt = minerva-today
    return dt.total_seconds()


def celcius_to_fahrenhate(c):
    return c*9/5 + 32


def miles_to_kilo(m):
    return 1.60934*m


print(get_minerva_year_seconds())
print(celcius_to_fahrenhate(23))
print(miles_to_kilo(5.1))

# print(zlib.decompress(base64.b64decode('eJxtUb1OwzAQ3v0U10xJi1AbUIdKhakClo5ICDG4ySW15Piq8xmat+eSpgyAdMPJ36/t1tPBevDUmvayxooYjTjxuM12Z+TKRYTV7Sozwv3GgGvyiBWFOj6sl3Od8n6el8vFnc6qKJQBHcZoW9yOLotsA09ENeyiuM6KozDLlDQGvY+Uj21pAH3Ef8VvlBhC6g7I4CLEznqvqxwxXI+pgakTuKDIQEM1EuQbqAkCCTTELQrYAyUZxJPihziWYjvc9tX6hDtm4nxqUxg8V3gS2NvugmjVP033Y0wK9Qye7SdCTwk4BdDCUKH3Eb6cHId8HCweNfESqMKX7uS1SRCsfyU3LqhBv9FPuj7XhJlvwyaZKQ==')))
# print(zlib.decompress(base64.b64decode('eJxtkE1PhDAQhu/9FbOcIKIRDDEhIZ426mVvejHGzJYBa4aW9GOFf2+B9eJ6m+SZeZ+37dkckYFNL/ptdNJYEl55pibZT2SlcgR3N0UivJ1rAapLWWlC+9EFLb0yOq3y4jYvs6apygx1e8Gvi7zK7yMvsxgAAzmHPTWr5Cqp4dGYFr7McZdEuhZ4W9l7UwogdvTv1YumaSTpqYUTcqAcRiaMZWNPwB6VXuIsLv1fl4W9tcam55xM0CRp9HDAYSNRcuE4GA+dCbrdwROeCGYTwAYNyAySmB18K/9pggdaIh6icRPGw+ch9hlIx4J/zJ3SMWCu47f/PvTMxA9Mn4h5')))
# print(zlib.decompress(base64.b64decode('eJx1kDFPAzEMhff8CjdTqzIA40knpgpYOsKAEErvfGmQE1eOU3r/ntxdWRBslt9730vsiQ+OgNgbv4y5Y0GjQQlbu7ugdCEj3FmjMjYGwgBx/OixbgOn9e2N1SDY203bWs+gDB25nG21QsScncd2hm1tA3ZrH5l7+OTDylbD3PU2y+/tvQGkjP8FX4WTh7OjgiCoRVJtrV5x0/tepv1OhGV9DW8MXjo8KexdXJRK/gu8Z4WBS+pX8OTOCCMXkJLAEUGHRBm+gh65KOBEeailS2cNPscTYcSk2P8qH0KqgLGpl/354FUz39pNfkQ=')))
print(zlib.decompress(base64.b64decode('eJx1kD1PwzAQhnf/iqunRGVpx0gRUwUsHWFACDnJm+Di+Cr7XJp/j5OUBcF2uvfj8Xlw3BhHjgc1rGNsOUCJFYdaH64IrY2gHRWCKPtSKwlTpcj2NE7vHbJq2Re7Oy02oNNlXWsxnyDjyTQRvoXOdhoRoxlQL8VbXZHe6gfmjk7cbHQ2LNzXRX6r94rgIv4LvgT2A12MS6AAScFncvYGM7/1ed4fQuBQ3MKlwrXFWehoxlXJzX8VH1mo5+S7DT2aC2jiRCHlS5yjFs5F+rLywUkIc8t9hq7MHHwazw4jvKD7Be+tzwVTlX/558Cbpr4BlL+CHg==')))