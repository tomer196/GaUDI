import warnings


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


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if variable.shape[-1] != 0:
            assert_correctly_masked(variable, node_mask)


def forward(x, h, node_mask, edge_mask, model):
    check_mask_correct([x, h], node_mask)
    assert_mean_zero_with_mask(x, node_mask)

    bs, n_nodes, n_dims = x.size()
    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
    xh = torch.cat([x, h], dim=-1)
    return model(xh, node_mask, edge_mask)


class RandomRotation(object):
    def __call__(self, x):
        d = x.shape[-1]
        # M = torch.randn(d, d).to(x.device)
        M = torch.randn(3, 3).to(x.device)
        I = torch.eye(3).to(x.device)
        M = torch.block_diag(I, M)
        Q, __ = torch.linalg.qr(M)
        return x @ Q


@torch.no_grad()
def main(args):
    # Prepare data
    # train_loader, val_loader, test_loader = create_data_loaders(args)
    # dataset = train_loader.dataset
    #
    # x, node_mask, edge_mask, node_features, y = next(iter(train_loader))
    # x = x.to(args.device)
    # node_mask = node_mask.to(args.device).unsqueeze(2)
    # edge_mask = edge_mask.to(args.device)
    # h = node_features.to(args.device)

    b = 5
    n = 10
    d = 6
    nf = 6
    model = EGNN_predictor(
        in_nf=nf,
        device=args.device,
        hidden_nf=args.nf,
        out_nf=1,
        act_fn=torch.nn.SiLU(),
        n_layers=args.n_layers,
        recurrent=True,
        tanh=args.tanh,
        attention=args.attention,
        d=d,
    )
    x = torch.randn(b, n, d, device=args.device)
    node_mask = torch.ones(b, n, 1, device=args.device)
    edge_mask = (
        torch.randn(b, n, n, device=args.device)
        - torch.eye(n, device=args.device)[None, :, :]
    )
    h = torch.randn(b, n, nf, device=args.device)
    x = remove_mean_with_mask(x, node_mask)

    rot = RandomRotation()
    o1 = forward(x, h, node_mask, edge_mask, model)
    o2 = forward(rot(x), h, node_mask, edge_mask, model)
    print("Rotation: ", torch.allclose(o1, o2, atol=1e-6))


if __name__ == "__main__":
    args = PredictionArgs().parse_args()
    args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    main(args)
