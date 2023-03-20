import json
import random
from time import time, sleep
import os
import warnings
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil

from edm.utils import gradient_clipping, Queue
from models_edm import get_model
from edm.equivariant_diffusion.utils import (
    remove_mean_with_mask,
    assert_correctly_masked,
    assert_mean_zero_with_mask,
)
from data.aromatic_dataloader import create_data_loaders
from sampling_edm import save_and_sample_chain_edm, sample_different_sizes_and_save_edm

from utils.args_edm import Args_EDM

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import torch

from torch import optim


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if variable.shape[-1] != 0:
            assert_correctly_masked(variable, node_mask)


def compute_loss_and_nll(model, nodes_dist, x, h, node_mask, edge_mask, context, args):
    bs, n_nodes, n_dims = x.size()
    assert_correctly_masked(x, node_mask)
    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

    h = {"categorical": h, "integer": torch.zeros(0).to(x.device)}

    nll = model(x, h, node_mask, edge_mask, context)

    # N = node_mask.squeeze(2).sum(1).long()
    # if args.orientation:
    #     N = torch.div(N, 2, rounding_mode="trunc")

    # log_pN = nodes_dist.log_prob(N)

    # assert nll.size() == log_pN.size()
    # nll = nll - log_pN

    # Average over batch.
    nll = nll.mean(0)

    return nll


def train_epoch(
    epoch, model, nodes_dist, dataloader, optimizer, args, writer, gradnorm_queue
):
    model.train()

    start_time = time()
    nll_losses = []
    grad_norms = []
    with tqdm(dataloader, unit="batch", desc=f"Train {epoch}") as tepoch:
        for i, (x, node_mask, edge_mask, node_features, y) in enumerate(tepoch):
            x = x.to(args.device)
            node_mask = node_mask.to(args.device).unsqueeze(2)
            edge_mask = edge_mask.to(args.device)
            h = node_features.to(args.device)

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, h], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            if args.conditioning:
                context = (
                    y[:, None, :].repeat(1, x.shape[1], 1).to(args.device) * node_mask
                )
            else:
                context = None

            # transform batch through flow
            nll = compute_loss_and_nll(
                model, nodes_dist, x, h, node_mask, edge_mask, context, args
            )

            # standard nll from forward KL
            loss = nll

            # backprop
            optimizer.zero_grad()
            loss.backward()

            if args.clip_grad:
                grad_norm = gradient_clipping(model, gradnorm_queue)
                grad_norms.append(grad_norm.item())

            optimizer.step()

            nll_losses.append(nll.item())
            tepoch.set_postfix(loss=np.mean(nll_losses).item())
    sleep(0.01)
    print(
        f"[{epoch}|train] nll loss: {np.mean(nll_losses):.3f}+-{np.std(nll_losses):.3f}, "
        f"GradNorm: {np.mean(grad_norms):.1f}, "
        f" in {int(time()-start_time)} secs"
    )
    writer.add_scalar("Train NLL", np.mean(nll_losses), epoch)
    writer.add_scalar("Train grad norm", np.mean(grad_norms), epoch)


def val_epoch(tag, epoch, model, nodes_dist, prop_dist, dataloader, args, writer):
    model.eval()
    with torch.no_grad():
        start_time = time()
        nll_losses = []
        with tqdm(dataloader, unit="batch", desc=f"{tag} {epoch}") as tepoch:
            for i, (x, node_mask, edge_mask, node_features, y) in enumerate(tepoch):
                x = x.to(args.device)
                node_mask = node_mask.to(args.device).unsqueeze(2)
                edge_mask = edge_mask.to(args.device)
                h = node_features.to(args.device)

                x = remove_mean_with_mask(x, node_mask)
                check_mask_correct([x, h], node_mask)
                assert_mean_zero_with_mask(x, node_mask)

                if args.conditioning:
                    context = (
                        y[:, None, :].repeat(1, x.shape[1], 1).to(args.device)
                        * node_mask
                    )
                else:
                    context = None

                # transform batch through flow
                nll = compute_loss_and_nll(
                    model, nodes_dist, x, h, node_mask, edge_mask, context, args
                )

                nll_losses.append(nll.item())

                tepoch.set_postfix(loss=np.mean(nll_losses).item())
        sleep(0.01)
        print(
            f"[{epoch}|{tag}] nll loss: {np.mean(nll_losses):.3f}+-{np.std(nll_losses):.3f}, "
            f" in {int(time() - start_time)} secs"
        )
        writer.add_scalar(f"{tag} NLL", np.mean(nll_losses), epoch)

        if tag == "val" and epoch % 50 == 0 and args.rings_graph:  # and epoch != 0:
            save_and_sample_chain_edm(
                args,
                model,
                prop_dist,
                dirname=f"{args.exp_dir}/epoch_{epoch}/",
                std=0.7,
            )
            sample_different_sizes_and_save_edm(
                args, model, nodes_dist, prop_dist, epoch=epoch, std=0.7
            )

    return np.mean(nll_losses)


def main(args):
    # Prepare data
    train_loader, val_loader, test_loader = create_data_loaders(args)

    # Choose model
    model, nodes_dist, prop_dist = get_model(args, train_loader)

    # Optimizer settings
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-12, amsgrad=True
    )
    gradnorm_queue = Queue(max_len=50)
    gradnorm_queue.add(3000)  # Add large value that will be flushed.

    # Save path
    writer = SummaryWriter(log_dir=args.exp_dir)

    # Run training
    print("-" * 20)
    print("Begin training")
    best_val_loss = 1e9
    best_epoch = 0
    for epoch in range(args.num_epochs):
        train_epoch(
            epoch,
            model,
            nodes_dist,
            train_loader,
            optimizer,
            args,
            writer,
            gradnorm_queue,
        )
        val_loss = val_epoch(
            "val", epoch, model, nodes_dist, prop_dist, val_loader, args, writer
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), args.exp_dir + "/model.pt")
        if epoch % 50 == 0 and epoch != 0:
            shutil.copy(args.exp_dir + "/model.pt", args.exp_dir + f"/model{epoch}.pt")

    print(f"{best_epoch=}, {best_val_loss=:.4f}")
    model.load_state_dict(torch.load(args.exp_dir + "/model.pt"))
    _ = val_epoch(
        "test", epoch, model, nodes_dist, prop_dist, test_loader, args, writer
    )
    writer.close()


if __name__ == "__main__":
    args = Args_EDM().parse_args()
    args.exp_dir = f"{args.save_dir}/{args.name}"
    print(args.exp_dir)

    # Create model directory
    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    with open(args.exp_dir + "/args.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)
    # Automatically choose GPU if available
    args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    print("Args:", args)

    # Where the magic is
    # with torch.autograd.detect_anomaly():
    main(args)
