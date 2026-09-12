from data_reader import df21, df22, df1
from lp_model import LPModel
import pandas as pd
import numpy as np


if __name__ == "__main__":
    data_L, data_P = df21.to_numpy(), df22.to_numpy()
    prices, L, solar = (
        np.tile(df1.to_numpy()[:, 1], 344),
        data_L[:, 1:].flatten(),
        data_P[:, 1:].flatten,
    )

    model = LPModel()

    x, fun = model(prices, L, solar, 6000, 6000)
    print(fun)
    df1 = pd.read_excel("./data/C/附件5/result1.xlsx")

    # df1["购电量"] = x[:144]
    # df1.to_excel("./res/result1.xlsx")
