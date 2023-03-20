import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def opv_data():
    opv_csv_file = "/home/tomerweiss/PBHs-design/data/db-474K-OPV-phase-2-filtered.csv"
    opv_df = pd.read_csv(
        opv_csv_file,
        usecols=[
            "HOMO",
            # "HOMO-LUMO gap/eV",
            # "Scharber_Voc_1",
            # "Scharber_Voc_2",
            # "Alharbi_Voc",
            # "Alharbi_FF",
            # "Scharber_PCE_1",
            # "Scharber_PCE_2",
            # "modified_Alharbi_PCE",
            # "atomisation energy per electron in kcal/mol",
            "HOMO-LUMO gap/eV",
            "electron_affinity e/V",
            "ionization e/V",
            "reorganisation energy eV",
            "oxidation potential eV",
        ],
    )
    # print(opv_df.loc[opv_df["modified_Alharbi_PCE"].idxmax()])

    # opv_df = opv_df[opv_df["HOMO-LUMO gap/eV"] > 0.5]
    # opv_df = opv_df[opv_df.modified_Alharbi_PCE > -100]
    # opv_df = opv_df[opv_df.modified_Alharbi_PCE < 100]

    property = "reorganisation energy eV"
    sns.histplot(opv_df, x=property)
    plt.title(property)
    plt.show()

    property = "oxidation potential eV"
    sns.histplot(opv_df, x=property)
    plt.title(property)
    plt.show()

    for p in [
        "electron_affinity e/V",
        "ionization e/V",
        "reorganisation energy eV",
        "oxidation potential eV",
    ]:
        g = sns.relplot(
            data=opv_df,
            x="HOMO",
            y=p,
        )
        g.fig.subplots_adjust(top=0.95)
        g.ax.set_title(f"HOMO Vs {p}")
        plt.show()

    for p in [
        "electron_affinity e/V",
        "ionization e/V",
        "reorganisation energy eV",
        "oxidation potential eV",
    ]:
        g = sns.relplot(
            data=opv_df,
            x="HOMO-LUMO gap/eV",
            y=p,
        )
        g.fig.subplots_adjust(top=0.95)
        g.ax.set_title(f"Gap Vs {p}")
        plt.show()


if __name__ == "__main__":
    opv_data()
