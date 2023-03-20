import json
import random
from time import time
import warnings

from torch.nn.functional import l1_loss

from data.aromatic_dataloader import create_data_loaders, AromaticDataset
from edm.equivariant_diffusion.utils import (
    assert_correctly_masked,
    remove_mean_with_mask,
    assert_mean_zero_with_mask,
)
from prediction.prediction_args import PredictionArgs
from prediction.train_predictor import get_prediction_model

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import torch
from torch import optim, Tensor
import matplotlib.pyplot as plt


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if variable.shape[-1] != 0:
            assert_correctly_masked(variable, node_mask)


def compute_loss(model, x, h, node_mask, edge_mask, target) -> Tensor:
    bs, n_nodes, n_dims = x.size()

    assert_correctly_masked(x, node_mask)

    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

    pred = model(x, node_mask, edge_mask)
    loss = l1_loss(pred, target)

    return loss


def val_epoch(tag, model, dataloader, args):
    model.eval()
    with torch.no_grad():
        start_time = time()
        loss_list = []
        rl_loss = []
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

            loss = compute_loss(model, x, h, node_mask, edge_mask, y)

            loss_list.append(loss.item())
            rl_loss.append(dataloader.dataset.rescale_loss(loss).item())

            # tepoch.set_postfix(loss=np.mean(loss_list).item())
        print(
            f"[{tag}] loss: {np.mean(loss_list):.4f}+-{np.std(loss_list):.4f}, "
            f"L1 (rescaled): {np.mean(rl_loss):.4f}, "
            f" in {int(time() - start_time)} secs"
        )

    return np.mean(loss_list)


def main(args):
    # Prepare data
    args.augmentation = False
    train_loader, val_loader, test_loader = create_data_loaders(args)
    model = get_prediction_model(args, train_loader.dataset)

    # Run training
    val_epoch("val", model, val_loader, args)
    val_epoch("test", model, test_loader, args)

    args.augmentation = True
    noise_list = torch.logspace(-3, -0.5, 10)
    val_losses = []
    for noise_std in noise_list:
        args.noise_std = noise_std.item()
        train_loader, val_loader, test_loader = create_data_loaders(args)
        print(noise_std)
        loss = val_epoch("test", model, test_loader, args)
        val_losses.append(loss * train_loader.dataset.std.mean().item())

    plt.plot(noise_list, val_losses)
    plt.xscale("log")
    plt.savefig(f"{args.exp_dir}/nosie_val.png")
    plt.show()
    np.save(f"{args.exp_dir}/nosie_val.npy", val_losses)


if __name__ == "__main__":
    args = PredictionArgs().parse_args()
    args.name = "test_1e-0.5"
    args.exp_dir = f"{args.save_dir}/predictor/{args.name}"
    print(args.exp_dir)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    with open(args.exp_dir + "/args.txt", "r") as f:
        args.__dict__ = json.load(f)
    args.restore = True

    # Automatically choose GPU if available
    args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    print("\n\nArgs:", args)

    # Where the magic is
    main(args)
