import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

df1 = pd.read_excel("./data/C/附件1.xlsx")  # 典型天
df22 = pd.read_excel("./data/C/附件2.xlsx", sheet_name="光伏发电实际功率")
df21 = pd.read_excel("./data/C/附件2.xlsx", sheet_name="小区负载")
df3 = pd.read_excel("./data/C/附件3.xlsx")  # 天气预测
df4 = pd.read_excel("./data/C/附件4.xlsx")  # 价格实际波动

dst = "./figures/"


def steady():
    data = df22.to_numpy()[:, 1:]

    data = data.flatten()[:343].reshape(-1, 7).sum(axis=0)

    # print(adfuller(data))

    # data = data[1:] - data[:-1]

    adf = adfuller(data)

    print(adf)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # plot_acf(data, lags=40, ax=ax1)  # 图上看截尾/拖尾
    # plot_pacf(data, lags=40, method="ywm", ax=ax2)

    plt.plot(data)

    plt.show()


def see():
    data = df21.to_numpy()[:, 1:]

    data = data[:343].reshape(-1, 7)
    for row in data:
        plt.scatter(np.arange(len(row)), row, s=4)

    plt.show()


if __name__ == "__main__":
    # steady()
    see()
