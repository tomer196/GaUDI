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


def compute_loss(model, x, h, node_mask, edge_mask):
    # helper function to compute loss - reshape inputs, run forward pass and return the loss
    bs, n_nodes, n_dims = x.size()
    assert_correctly_masked(x, node_mask)
    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

    h = {"categorical": h, "integer": torch.zeros(0).to(x.device)}

    loss = model(x, h, node_mask, edge_mask)

    # Average over batch.
    loss = loss.mean(0)

    return loss


def train_epoch(epoch, model, dataloader, optimizer, args, writer, gradnorm_queue):
    model.train()

    start_time = time()
    losses = []
    grad_norms = []
    with tqdm(dataloader, unit="batch", desc=f"Train {epoch}") as tepoch:
        for i, (x, node_mask, edge_mask, node_features, y) in enumerate(tepoch):
            # prepare data - send to device, reshape, etc.
            x = x.to(args.device)
            node_mask = node_mask.to(args.device).unsqueeze(2)
            edge_mask = edge_mask.to(args.device)
            h = node_features.to(args.device)

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, h], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            # forward pass
            loss = compute_loss(model, x, h, node_mask, edge_mask)

            # backprop
            optimizer.zero_grad()
            loss.backward()

            if args.clip_grad:
                grad_norm = gradient_clipping(model, gradnorm_queue)
                grad_norms.append(grad_norm.item())

            optimizer.step()

            losses.append(loss.item())
            tepoch.set_postfix(loss=np.mean(losses).item())
    sleep(0.01)

    # print and log results
    print(
        f"[{epoch}|train] loss: {np.mean(losses):.3f}+-{np.std(losses):.3f}, "
        f"GradNorm: {np.mean(grad_norms):.1f}, "
        f" in {int(time()-start_time)} secs"
    )
    writer.add_scalar("Train loss", np.mean(losses), epoch)
    writer.add_scalar("Train grad norm", np.mean(grad_norms), epoch)


def val_epoch(tag, epoch, model, nodes_dist, prop_dist, dataloader, args, writer):
    model.eval()
    with torch.no_grad():
        start_time = time()
        losses = []
        with tqdm(dataloader, unit="batch", desc=f"{tag} {epoch}") as tepoch:
            for i, (x, node_mask, edge_mask, node_features, y) in enumerate(tepoch):
                # prepare data - send to device, reshape, etc.
                x = x.to(args.device)
                node_mask = node_mask.to(args.device).unsqueeze(2)
                edge_mask = edge_mask.to(args.device)
                h = node_features.to(args.device)

                x = remove_mean_with_mask(x, node_mask)
                check_mask_correct([x, h], node_mask)
                assert_mean_zero_with_mask(x, node_mask)

                # forward pass
                loss = compute_loss(model, x, h, node_mask, edge_mask)

                losses.append(loss.item())

                tepoch.set_postfix(loss=np.mean(losses).item())
        sleep(0.01)

        # print and log results
        print(
            f"[{epoch}|{tag}] loss: {np.mean(losses):.3f}+-{np.std(losses):.3f}, "
            f" in {int(time() - start_time)} secs"
        )
        writer.add_scalar(f"{tag} loss", np.mean(losses), epoch)

        # save samples
        if tag == "val" and epoch % 50 == 0 and args.rings_graph:
            save_and_sample_chain_edm(
                args,
                model,
                dirname=f"{args.exp_dir}/epoch_{epoch}/",
                std=0.7,
            )
            sample_different_sizes_and_save_edm(
                args, model, nodes_dist, prop_dist, epoch=epoch, std=0.7
            )

    return np.mean(losses)


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

    # Create logger
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
            train_loader,
            optimizer,
            args,
            writer,
            gradnorm_queue,
        )
        val_loss = val_epoch(
            "val", epoch, model, nodes_dist, prop_dist, val_loader, args, writer
        )
        # save best model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), args.exp_dir + "/model.pt")

    # load best model and evaluate on the test set
    print(f"{best_epoch=}, {best_val_loss=:.4f}")
    model.load_state_dict(torch.load(args.exp_dir + "/model.pt"))
    _ = val_epoch(
        "test", epoch, model, nodes_dist, prop_dist, test_loader, args, writer
    )
    writer.close()


if __name__ == "__main__":
    # set seeds
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Load arguments and save in the experiment directory
    args = Args_EDM().parse_args()
    args.exp_dir = f"{args.save_dir}/{args.name}"

    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    with open(args.exp_dir + "/args.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # Automatically choose GPU if available
    args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    print(args.exp_dir)
    print("Args:", args)

    # Run training
    main(args)
