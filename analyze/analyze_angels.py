import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch import quantile, Tensor

dataset = "hetro"


def analyze_dists():
    print("Bond dists")
    # angels = torch.load(f'analyze/dists_{dataset}.pt')
    # df = pd.DataFrame(angels, columns=['from', 'to', 'dist']).astype(float)
    df = pd.read_pickle(f"analyze/dists_{dataset}.pkl")
    df["reverse"] = df["from"] > df["to"]
    df["bond"] = 0

    # knots = RINGS_LIST[dataset]
    # mapping = {i: k for i, k in enumerate(knots)}
    # df['from'] = df['from'].astype(int).map(mapping)
    # df['to'] = df['to'].astype(int).map(mapping)
    df.loc[df["reverse"], "bond"] = df["from"] + "-" + df["to"]
    df.loc[~df["reverse"], "bond"] = df["to"] + "-" + df["from"]

    for bond in df["bond"].unique():
        if bond is None:
            continue
        df_bond = df[df["bond"] == bond]["dist"]
        q_lo, q_hi = df_bond.quantile([0.01, 0.99])
        print(f"'{bond}': ({q_lo:.2f}, {q_hi:.2f}),")

        # title = f'{bond}: {len(df_bond)}, {q_lo=:.2f}, {q_hi=:.2f}'
        # print(title)
        # g = sns.displot(df_bond)
        # g.fig.subplots_adjust(top=.95)
        # plt.vlines([q_lo, q_hi], plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], colors='r')
        # plt.title(title)
        # plt.show()


def analyze_angels3():
    print("angels 3")
    df = pd.read_pickle(f"analyze/angels3_{dataset}.pkl")
    sns.histplot(torch.stack(list(df.angel.values)))
    plt.title("All angels")
    plt.show()
    for symbol in df["symbol"].unique():
        if symbol is None:
            continue
        df_symbol = df[df["symbol"] == symbol]["angel"]
        title = f"{symbol}: {len(df_symbol)}"
        print(f"'{symbol}': " + "{")

        sns.histplot(torch.stack(list(df_symbol.values)))
        plt.xlim([100, 190])
        plt.title(title)
        plt.show()
        angels = torch.stack(list(df_symbol.values))
        for k in [120, 140, 180]:
            a = angels[torch.logical_and(k - 25 < angels, angels < k + 25)]
            if a.shape[0] != 0:
                print(f"   '{k}': ({quantile(a, 0.001)}, {quantile(a, 0.999)}),")
        print("},")


def analyze_angels4():
    print("angles 4")
    df = pd.read_pickle(f"analyze/angels4_{dataset}.pkl")
    df.angel = df.angel.apply(lambda x: x.item())
    # df = df[df['1']!='Cbd'][df['2']!='Cbd'][df['3']!='Cbd'][df['4']!='Cbd']
    df1 = df[df.angel < 100]
    df2 = df1[df1.angel > 80]
    angels = torch.stack(list(df.angel.values))
    # a = angels[angels<60]
    # print(f"'0': {quantile(a, 0.99)},")
    # a = angels[angels>120]
    # print(f"'180': {quantile(a, 0.01)},")

    sns.histplot(angels)
    plt.show()


if __name__ == "__main__":
    # analyze_dists()
    # analyze_angels3()
    analyze_angels4()
