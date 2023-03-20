import json
import random
from time import time, sleep
import warnings
import os

import matplotlib.pyplot as plt

from edm.equivariant_diffusion.utils import (
    remove_mean_with_mask,
    assert_mean_zero_with_mask,
)

from data.aromatic_dataloader import create_data_loaders
from models_edm import get_model
from cond_prediction.prediction_args import PredictionArgs
from cond_prediction.train_cond_predictor import (
    get_cond_predictor_model,
    check_mask_correct,
    compute_loss,
)
from utils.args_edm import Args_EDM

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import torch
import os

from torch import linspace


def val_epoch(tag, cond_predictor, edm_model, dataloader, args, edm_args, t_fix=None):
    cond_predictor.eval()
    with torch.no_grad():
        start_time = time()
        loss_list = []
        rl_loss = []
        error_list = []
        # with tqdm(dataloader, unit="batch", desc=f"{tag} {epoch}") as tepoch:
        for i, (x, node_mask, edge_mask, node_features, y) in enumerate(dataloader):
            x = x.to(args.device)
            y = y.to(args.device)
            node_mask = node_mask.to(args.device).unsqueeze(2)
            edge_mask = edge_mask.to(args.device)
            h = node_features.to(args.device)

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, h], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            loss, err = compute_loss(
                cond_predictor,
                x,
                h,
                node_mask,
                edge_mask,
                y,
                edm_model,
                edm_args,
                t_fix,
            )
            error_list.append(err)
            loss_list.append(loss.item())
            rl_loss.append(dataloader.dataset.rescale_loss(loss).item())

            # tepoch.set_postfix(loss=np.mean(loss_list).item())
        print(
            f"[{tag}] loss: {np.mean(loss_list):.4f}+-{np.std(loss_list):.4f}, "
            f"L1 (rescaled): {np.mean(rl_loss):.4f}, "
            f" in {int(time() - start_time)} secs"
        )
        std = dataloader.dataset.std
        # print(f"Error: {torch.cat(error_list).mean(0)}")
        # print(f"Error: {torch.cat(error_list).mean(0).cpu()*std}")
        print()
        sleep(0.01)

    return (torch.cat(error_list).cpu() * std[None, :]).mean().item()


def main(pred_args, edm_args):
    # Prepare data
    train_loader, val_loader, test_loader = create_data_loaders(pred_args)

    cond_predictor = get_cond_predictor_model(pred_args, train_loader.dataset)

    # EDM model
    edm_model, nodes_dist, prop_dist = get_model(edm_args, train_loader)

    print("Test all times:")
    val_epoch("test", cond_predictor, edm_model, test_loader, pred_args, edm_args)
    losses = []
    times = linspace(0, edm_model.T, 11)
    for t_fix in times:
        print(f"Test time {t_fix}:")
        losses.append(
            val_epoch(
                "test",
                cond_predictor,
                edm_model,
                test_loader,
                pred_args,
                edm_args,
                t_fix=t_fix,
            )
        )

    plt.plot(times, losses)
    plt.xlabel("Timestamp (noise level)")
    plt.ylabel("MAE")
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    edm_args = Args_EDM().parse_args()
    pred_args = PredictionArgs().parse_args()
    pred_args.name = "cata-test"
    pred_args.exp_dir = f"{pred_args.save_dir}/{pred_args.name}"
    print(pred_args.exp_dir)

    # Create model directory
    if not os.path.isdir(pred_args.exp_dir):
        os.makedirs(pred_args.exp_dir)

    with open(pred_args.exp_dir + "/args.txt", "r") as f:
        pred_args.__dict__ = json.load(f)
    pred_args.restore = True

    # Automatically choose GPU if available
    pred_args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    edm_args.device = pred_args.device

    print("\n\nArgs:", pred_args)

    # Where the magic is
    main(pred_args, edm_args)
