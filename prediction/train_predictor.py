import json
import random
from time import time, sleep
import os
import warnings

from torch.nn.functional import l1_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from edm.egnn_predictor.models import EGNN_predictor
from edm.equivariant_diffusion.utils import (
    remove_mean_with_mask,
    assert_correctly_masked,
    assert_mean_zero_with_mask,
)
from data.aromatic_dataloader import create_data_loaders, AromaticDataset
from prediction.prediction_args import PredictionArgs

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import torch

from torch import optim, Tensor


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if variable.shape[-1] != 0:
            assert_correctly_masked(variable, node_mask)


def compute_loss(model, x, h, node_mask, edge_mask, target) -> Tensor:
    bs, n_nodes, n_dims = x.size()

    assert_correctly_masked(x, node_mask)

    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

    xh = torch.cat([x, h], dim=-1)
    pred = model(xh, node_mask, edge_mask)
    loss = l1_loss(pred, target)

    return loss


def train_epoch(epoch, model, dataloader, optimizer, args, writer):
    model.train()

    start_time = time()
    loss_list = []
    rl_loss = []
    with tqdm(dataloader, unit="batch", desc=f"Train {epoch}") as tepoch:
        for i, (x, node_mask, edge_mask, node_features, y) in enumerate(tepoch):
            x = x.to(args.device)
            y = y.to(args.device)
            node_mask = node_mask.to(args.device).unsqueeze(2)
            edge_mask = edge_mask.to(args.device)
            h = node_features.to(args.device)

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, h], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            loss = compute_loss(model, x, h, node_mask, edge_mask, y)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            rl_loss.append(dataloader.dataset.rescale_loss(loss).item())

            tepoch.set_postfix(loss=np.mean(loss_list).item())
    print(
        f"[{epoch}|train] loss: {np.mean(loss_list):.4f}+-{np.std(loss_list):.4f}, "
        f"L1 (rescaled): {np.mean(rl_loss):.4f}, "
        f" in {int(time()-start_time)} secs"
    )
    sleep(0.01)
    writer.add_scalar("Train loss", np.mean(loss_list), epoch)
    writer.add_scalar("Train L1 (rescaled)", np.mean(rl_loss), epoch)


def val_epoch(tag, epoch, model, dataloader, args, writer):
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
            f"[{epoch}|{tag}] loss: {np.mean(loss_list):.4f}+-{np.std(loss_list):.4f}, "
            f"L1 (rescaled): {np.mean(rl_loss):.4f}, "
            f" in {int(time() - start_time)} secs"
        )
        sleep(0.01)
        writer.add_scalar(f"{tag} loss", np.mean(loss_list), epoch)
        writer.add_scalar(f"{tag} L1 (rescaled)", np.mean(rl_loss), epoch)

    return np.mean(loss_list)


def get_prediction_model(args, dataset: AromaticDataset):
    model = EGNN_predictor(
        in_nf=dataset.num_node_features,
        device=args.device,
        hidden_nf=args.nf,
        out_nf=dataset.num_targets,
        act_fn=torch.nn.SiLU(),
        n_layers=args.n_layers,
        recurrent=True,
        tanh=args.tanh,
        attention=args.attention,
    )

    if args.restore is not None:
        model_state_dict = torch.load(args.exp_dir + "/model.pt")
        model.load_state_dict(model_state_dict)
    if args.dp and torch.cuda.device_count() > 1:
        print(f"Training using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    return model


def main(args):
    # Prepare data
    train_loader, val_loader, test_loader = create_data_loaders(args)

    model = get_prediction_model(args, train_loader.dataset)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, amsgrad=True, weight_decay=1e-12
    )

    # Save path
    writer = SummaryWriter(log_dir=args.exp_dir)

    # Run training
    print("Begin training")
    best_val_loss = 1e9
    best_epoch = 0
    for epoch in range(args.num_epochs):
        train_epoch(epoch, model, train_loader, optimizer, args, writer)
        val_loss = val_epoch("val", epoch, model, val_loader, args, writer)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), args.exp_dir + "/model.pt")

    print(f"{best_epoch=}, {best_val_loss=:.4f}")
    model.load_state_dict(torch.load(args.exp_dir + "/model.pt"))
    _ = val_epoch("test", epoch, model, test_loader, args, writer)
    writer.close()


if __name__ == "__main__":
    args = PredictionArgs().parse_args()
    args.exp_dir = f"{args.save_dir}/predictor/{args.name}"
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
    main(args)
