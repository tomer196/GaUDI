import os
from pathlib import Path

import pandas as pd


def combine_cata_peri():
    cata_csv_file = "/home/tomerweiss/PBHs-design/data/COMPAS-1x.csv"
    peri_csv_file = "/home/tomerweiss/PBHs-design/data/peri-xtb-data-55821.csv"
    cata_df = pd.read_csv(cata_csv_file)
    peri_df = pd.read_csv(peri_csv_file)
    print(len(cata_df), cata_df.columns)
    print(len(peri_df), peri_df.columns)

    cols = cata_df.columns.intersection(peri_df.columns)
    print(cols)
    df_all = pd.concat([cata_df[cols], peri_df[cols]]).reset_index(drop=True)
    print(len(df_all), df_all.columns)
    df_all.to_csv(f"/home/tomerweiss/PBHs-design/data/cata-peri-{len(df_all)}.csv")


def opv_data():
    hetro_csv_file = "/home/tomerweiss/PBHs-design/data/db-474K-filtered.csv"
    opv_csv_file = "/home/tomerweiss/PBHs-design/data/db-474K-OPV-phase-2.csv"
    hetro_df = pd.read_csv(hetro_csv_file, usecols=["name"])
    opv_df = pd.read_csv(opv_csv_file)
    print(len(hetro_df), hetro_df.columns)
    print(len(opv_df), opv_df.columns)
    print(opv_df.iloc[0])
    opv_df = opv_df[opv_df["name"].isin(hetro_df["name"])].reset_index(drop=True)
    print(len(opv_df), opv_df.columns)
    print("mol" in opv_df.columns)
    opv_df.to_csv("/home/tomerweiss/PBHs-design/data/db-474K-OPV-phase-2-filtered.csv")


if __name__ == "__main__":
    opv_data()
