import numpy as np
from data_reader import df1
from scipy.optimize import linprog


class LPModel:
    def __init__(self):
        self.E_up = 10800.0
        self.E_down = 1200.0

        self.P = 5000.0
        self.eta = 0.9

    def __call__(self, prices, L, solar, E0, Ee=None):
        "传入 价格 小区载荷 , 光伏发电 , 起始值和终点值"
        T = len(prices)

        def _idx(m, n):
            return m * T + n

        C = np.hstack([prices, np.zeros(T * 3)])

        A_uq = np.zeros((T, T * 4))

        for t in range(T):
            A_uq[t, _idx(2, t)], A_uq[t, _idx(1, t)], A_uq[t, _idx(0, t)] = (
                1.0,
                -1.0,
                -1.0,
            )

        B_uq = solar - L

        A_eq = np.zeros((T + 1, T * 4))
        for t in range(T):
            A_eq[t, _idx(3, t)] = 1.0
            if t:
                A_eq[t, _idx(3, t - 1)] = -1.0

            A_eq[t, _idx(1, t)] = 1 / self.eta
            A_eq[t, _idx(2, t)] = -self.eta

        A_eq[T, _idx(3, T - 1)] = 1.0 if Ee is not None else 0.0
        B_eq = np.zeros(T + 1)
        B_eq[0], B_eq[T] = E0, Ee if Ee is not None else 0.0

        bounds = (
            [(0, None) for _ in range(T)]
            + [(0, self.P / 6) for _ in range(T)]
            + [(0, self.P / 6) for _ in range(T)]
            + [(self.E_down, self.E_up) for _ in range(T)]
        )

        res = linprog(C, A_ub=A_uq, b_ub=B_uq, A_eq=A_eq, b_eq=B_eq, bounds=bounds)

        return res.x, res.fun


if __name__ == "__main__":
    data = df1.to_numpy()
    prices, L, solar = data[:, 1], data[:, 2] / 6, data[:, 3] / 6

    model = LPModel()

    x, fun = model(prices, L, solar, 6000, 6000)

    print(fun)
