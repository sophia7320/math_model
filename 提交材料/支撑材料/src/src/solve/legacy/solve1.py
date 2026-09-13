from data_reader import df1
from lp_model import LPModel
import pandas as pd


if __name__ == "__main__":
    data = df1.to_numpy()
    prices, L, solar = data[:, 1], data[:, 2] / 6, data[:, 3] / 6

    model = LPModel()

    x, fun = model(prices, L, solar, 6000, 6000)
    print(fun)
    df1 = pd.read_excel("./data/C/附件5/result1.xlsx")

    df1["购电量"] = x[:144]
    df1.to_excel("./res/result1.xlsx")
