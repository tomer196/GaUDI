import torch

from data.aromatic_dataloader import create_data_loaders
from utils.args_edm import Args_EDM
from utils.plotting import (
    plot_graph_of_rings,
    plot_graph_of_atoms,
    plot_rings_orientation,
    plot_rdkit,
)

cata1_names = [
    "hc_c22h14_0pent_12",  # LLL
    "hc_c46h26_0pent_25798",
    "hc_c46h26_0pent_25603",
    "hc_c46h26_0pent_25589",
    "hc_c46h26_0pent_25705",
    "hc_c46h26_0pent_25801",
    "hc_c46h26_0pent_25448",
    "hc_c46h26_0pent_25272",
    "hc_c46h26_0pent_25706",
    "hc_c46h26_0pent_25271",
    "hc_c46h26_0pent_25728",
]
hetro_gaps = ["C2M253821", "C2M067283", "C2M265611"]
hetro_multi = ["C2M254157", "C2M389597", "C2M410435"]
temp = ["C2M003651", "C2M006590"]
names = temp

rings = True

args = Args_EDM().parse_args()
args.dataset = "hetro"
args.orientation = args.dataset == "hetro"
orientation = args.dataset == "hetro"
if args.dataset == "hetro":
    args.target_features = "HOMO-LUMO gap/eV,HOMO,electron_affinity e/V,ionization e/V"
else:
    args.target_features = "LUMO_eV,GAP_eV,Erel_eV,aIP_eV,aEA_eV"
train_loader, _, _ = create_data_loaders(args)
dataset = train_loader.dataset
dataset.orientation = orientation
dataset.return_adj = True
df = args.df_all
for name in names:
    df_row = df.loc[df["molecule"] == name].squeeze()
    value = df_row[args.target_features.split(",")].values
    value = [f"{v:.3f}" for v in value]
    if rings:
        x, node_mask, edge_mask, node_features, adj, y = dataset.get_all(df_row)
        x = x[node_mask.bool()]
        node_features = node_features[node_mask.bool()]
        filename = "/home/tomerweiss/PBHs-design/mols/" + name + ".pdf"
        # plot_graph_of_rings(
        #     x,
        #     node_features.argmax(1),
        #     filename=filename,
        #     title=f"{name}\n{value:.3f}",
        #     tol=0.1,
        #     showPlot=True,
        #     dataset="hetro" if rings else "cata",
        #     orientation=orientation,
        # )
        plot_rdkit(
            x,
            node_features.argmax(1),
            filename=filename,
            tol=0.1,
            title=f"{name}\n{value}",
            dataset=args.dataset,
        )
    else:
        x, adj, node_features = dataset.get_atoms(df_row)
        filename = "/home/tomerweiss/PBHs-design/mols/" + name + "_atoms.png"
        plot_graph_of_atoms(
            x,
            node_features,
            adj,
            filename=filename,
            title=f"{name}\n{value:.3f}",
            showPlot=True,
            tol=0.1,
            axis_lim=15,
        )

    # if orientation:
    #     x, node_mask, edge_mask, node_features, adj, _ = dataset.get_all(df_row)
    #     filename = "/home/tomerweiss/PBHs-design/mols/" + name + "_orientation.png"
    #     plot_rings_orientation(
    #         x,
    #         node_mask,
    #         node_features.argmax(1),
    #         adj,
    #         filename=filename,
    #         title=f"{name}\n{value:.3f}",
    #         tol=0.1,
    #         showPlot=True,
    #     )

    print(f"{name} done")
