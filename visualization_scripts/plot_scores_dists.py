import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import font_manager

font_dirs = ["/home/tomerweiss/.fonts/computer-modern/"]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

# set font
plt.rcParams["font.family"] = "CMU Serif"


plt.rcParams.update({"font.size": 18})


def cata1():
    palette = plt.get_cmap("winter")
    cmap = plt.cm.colors.ListedColormap([palette(x) for x in np.linspace(0, 1, 5)])
    colors = ["black"] + cmap.colors[::-1]
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for i, name in enumerate(["data", 0, 10, 30, 100, 1000]):
        scores = np.load(f"visualization_scripts/cata1/scores_{name}.npy")
        if name == "data":
            sns.kdeplot(scores, ax=ax, color=colors[i], fill=True, alpha=0.5)
        else:
            sns.kdeplot(scores, ax=ax, color=colors[i], fill=True)
    ax.set_xlabel("Target function value")
    ax.set_xlim([0, 13])
    # ax.set_ylim([0, 18])
    ax.set_ylabel(None)
    ax.set_yticks([])
    # ax.set_xticks([-0.1, 0, 0.1, 0.2])
    ax.legend(
        [
            "Dataset",
            "Unconditional (s=0)",
            "Conditional (s=10)",
            "Conditional (s=30)",
            "Conditional (s=100)",
            "Conditional (s=1000)",
        ],
        loc="upper right",
    )
    plt.savefig(
        "visualization_scripts/cata1/scores_cata1.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )
    fig.show()


def hetro_gap():
    palette = plt.get_cmap("winter")
    cmap = plt.cm.colors.ListedColormap([palette(x) for x in np.linspace(0, 1, 6)])
    colors = ["black"] + cmap.colors[::-1]
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for i, name in enumerate(["data", 0, 10, 30, 100, 300, 1000]):
        scores = np.load(f"visualization_scripts/hetro_gap/scores_{name}.npy")
        gap = -scores
        if name == "data":
            sns.kdeplot(gap, ax=ax, color=colors[i], fill=True, alpha=0.5)
        else:
            sns.kdeplot(gap, ax=ax, color=colors[i], fill=True)
    ax.set_xlabel("HOMO-LUMO gap (eV)")
    ax.set_xlim([0, 3])
    # ax.set_ylim([0, 18])
    ax.set_ylabel(None)
    ax.set_yticks([])
    # ax.set_xticks([-0.1, 0, 0.1, 0.2])
    ax.legend(
        [
            "Dataset",
            "Unconditional (s=0)",
            "Conditional (s=10)",
            "Conditional (s=30)",
            "Conditional (s=100)",
            "Conditional (s=1000)",
        ],
        loc="upper left",
    )
    plt.savefig(
        "visualization_scripts/hetro_gap/scores_hetro_gap.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )
    fig.show()


def hetro_multi():
    palette = plt.get_cmap("winter")
    cmap = plt.cm.colors.ListedColormap([palette(x) for x in np.linspace(0, 1, 5)])
    colors = ["black"] + cmap.colors[::-1]
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for i, name in enumerate(["data", 0, 10, 30, 100, 300]):
        scores = np.load(f"visualization_scripts/hetro_multi/scores_{name}.npy")
        if name == "data":
            sns.kdeplot(scores, ax=ax, color=colors[i], fill=True, alpha=0.5)
        else:
            sns.kdeplot(scores, ax=ax, color=colors[i], fill=True)
    ax.set_xlabel("Target function value")
    ax.set_xlim([1.5, 12.5])
    # ax.set_ylim([0, 18])
    ax.set_ylabel(None)
    ax.set_yticks([])
    # ax.set_xticks([-0.1, 0, 0.1, 0.2])
    ax.legend(
        [
            "Dataset",
            "Unconditional (s=0)",
            "Conditional (s=10)",
            "Conditional (s=30)",
            "Conditional (s=100)",
            "Conditional (s=300)",
        ],
        loc="upper right",
    )
    plt.savefig(
        "visualization_scripts/hetro_multi/scores_hetro_multi.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )
    fig.show()


if __name__ == "__main__":
    cata1()
    hetro_gap()
    hetro_multi()
