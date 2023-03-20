import json
import warnings
from typing import Tuple

from torch import Tensor, randn
from torch.nn.functional import l1_loss

from analyze.analyze import analyze_stability_for_molecules
from edm.equivariant_diffusion.utils import (
    assert_correctly_masked,
    assert_mean_zero_with_mask,
)
from models_edm import get_model
from prediction.eval_predictor import get_prediction_model
from prediction.prediction_args import PredictionArgs
from prediction.train_cond_predictor import get_cond_predictor_model
from sampling_edm import sample_pos_edm, sample_guidance
from data.aromatic_dataloader import create_data_loaders

from utils.args_edm import Args_EDM
from utils.plotting import plot_graph_of_rings

warnings.simplefilter(action="ignore", category=FutureWarning)

import torch


@torch.no_grad()
def predict(model, x, h, node_mask, edge_mask, edm_model) -> Tuple[Tensor, Tensor]:
    bs, n_nodes, n_dims = x.size()
    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
    node_mask = node_mask.view(bs, n_nodes, 1)

    t = torch.zeros(x.size(0), 1, device=x.device).float()
    x, h, _ = edm_model.normalize(
        x,
        {"categorical": h, "integer": torch.zeros(0).to(x.device)},
        node_mask,
    )
    xh = torch.cat([x, h["categorical"]], dim=-1)
    pred = model(xh, node_mask, edge_mask, t)
    return pred


def eval_stability(x, one_hot, dataset="cata", orientation=False):
    atom_type = one_hot.argmax(2).cpu().detach()
    molecule_list = [(x[i].cpu().detach(), atom_type[i]) for i in range(x.shape[0])]
    stability_dict, molecule_stable_list = analyze_stability_for_molecules(
        molecule_list, args.tol, dataset=dataset, orientation=orientation
    )
    x = x[stability_dict["molecule_stable_bool"]]
    atom_type = atom_type[stability_dict["molecule_stable_bool"]]
    return stability_dict, x, atom_type


def switch_grad_off(models):
    for m in models:
        for p in m.parameters():
            p.requires_grad = False


def design(args, model, cond_predictor, nodes_dist, prop_dist):
    switch_grad_off([model, cond_predictor])
    model.eval()
    cond_predictor.eval()

    args.batch_size = 256
    args.tol = 0.1

    n_nodes = 10
    nodesxsample = Tensor([n_nodes] * args.batch_size).long()
    # nodesxsample = nodes_dist.sample(args.batch_size)
    # nodesxsample = torch.arange(5, 11).repeat(200)

    # target = torch.Tensor([-7.646, 2.035, 0.854, 11.072, -6.24])
    target = torch.Tensor([-8.141, 1.23, 0.5, 11.254, -6.278])
    # target = None
    # target = torch.Tensor([2])
    if target is not None:
        context = target[None, :].repeat(nodesxsample.shape[0], 1)
        context = prop_dist.normalize(context)
    else:
        # context = prop_dist.sample_df(nodesxsample)
        context = prop_dist.sample_batch(nodesxsample)
        # print(prop_dist.unnormalize(context))

    if args.conditioning:
        print("\nDesign molecule - standard")
        x, one_hot, node_mask, edge_mask = sample_pos_edm(
            args, model, prop_dist, nodesxsample, std=0.7, context=context
        )
    else:
        print("\nDesign molecule - guidance")
        context = context.to(args.device)

        # def target_function(_input, _node_mask, _edge_mask, _t):
        #     pred = cond_predictor(_input, _node_mask, _edge_mask, _t)
        #     return (pred - context).abs().mean(1)

        def target_function(_input, _node_mask, _edge_mask, _t):
            pred = cond_predictor(_input, _node_mask, _edge_mask, _t)
            return (pred - context).abs()

        x, one_hot, node_mask, edge_mask = sample_guidance(
            args,
            model,
            target_function,
            nodesxsample,
            context=None,
            c_scale=100,
            # std=0.7,
        )

    assert_correctly_masked(x, node_mask)
    assert_mean_zero_with_mask(x, node_mask)

    stability_dict, _, _ = eval_stability(
        x, one_hot, dataset=args.dataset, orientation=args.orientation
    )
    print(f"{stability_dict['mol_stable']=} out of {x.shape[0]}")
    pred = (
        predict(cond_predictor, x, one_hot, node_mask, edge_mask, model).detach().cpu()
    )

    context = context.cpu()
    pred = prop_dist.unnormalize(pred)
    context = prop_dist.unnormalize(context)
    loss = l1_loss(pred, context)
    print(f"MAE: {loss.item():.4f}")

    pred = pred[stability_dict["molecule_stable_bool"]]
    context = context[stability_dict["molecule_stable_bool"]]
    loss = l1_loss(pred, context)
    print(f"MAE stable: {loss.item():.4f}")

    # if context.shape[1] == 1:
    #     best_idx = (context - pred).abs().mean(1).min(0).indices.item()
    #     best_value = (context - pred).abs()[best_idx]
    #     print(
    #         f"best value (from stable): {pred[best_idx]}, " f"loss: {best_value.mean()}"
    #     )
    #     plot_graph_of_rings(
    #         x[best_idx].detach().cpu(),
    #         None,
    #         filename=f"best{target}.png",
    #         tol=args.tol,
    #         title=f"{target}, {pred[best_idx]}",
    #     )

    # plt.hist(pred)
    # plt.savefig(
    #     f"{pred_args.name}-{target}.png", bbox_inches="tight", pad_inches=0.0
    # )


def main(args, cond_predictor_args):
    train_loader, _, test_loader = create_data_loaders(cond_predictor_args)
    model, nodes_dist, prop_dist = get_model(args, test_loader)
    cond_predictor = get_cond_predictor_model(cond_predictor_args, train_loader.dataset)

    design(args, model, cond_predictor, nodes_dist, prop_dist)


if __name__ == "__main__":
    args = Args_EDM().parse_args()
    # args.name = "cond/lumo_gap_erel_ip_ea"
    # args.name = "cond/lumo_gap_erel_ip_ea__final_act_tanh"
    # args.name = "EDM_6_192_range4"
    # args.name = "cond/lumo_gap_erel_ip_ea"

    # args.name = "cond/cata_l9_c196_polynomial_2_take2"
    args.name = "cond/cata_l9_c196_polynomial_2"
    args.name = "cata_l9_c196_polynomial_2"
    args.exp_dir = f"{args.save_dir}/{args.name}"
    print(args.exp_dir)

    with open(args.exp_dir + "/args.txt", "r") as f:
        args.__dict__ = json.load(f)
    args.restore = True

    cond_predictor_args = PredictionArgs().parse_args()
    # cond_predictor_args.name = (
    #     "cond_predictor/hetro_gap_homo_ea_ip_stability_polynomial_2_with_norm"
    # )
    cond_predictor_args.name = (
        "cond_predictor/cata_lumo_gap_erel_ip_ea_polynomial_2_with_norm"
    )
    cond_predictor_args.exp_dir = (
        f"{cond_predictor_args.save_dir}/{cond_predictor_args.name}"
    )
    print(cond_predictor_args.exp_dir)

    with open(cond_predictor_args.exp_dir + "/args.txt", "r") as f:
        cond_predictor_args.__dict__ = json.load(f)
    cond_predictor_args.restore = True

    # Automatically choose GPU if available
    args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    cond_predictor_args.device = args.device

    # Where the magic is
    main(args, cond_predictor_args)
