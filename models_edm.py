import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import numpy as np

from edm.egnn.models import EGNN_dynamics_QM9
from edm.equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from edm.egnn.egnn_new import EGNN, GNN
from edm.equivariant_diffusion.utils import remove_mean, remove_mean_with_mask
from utils.utils import analyzed_rings


class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        if name == "module":
            return super().__getattr__("module")
        else:
            return getattr(self.module, name)


class DistributionRings:
    def __init__(self, dataset="cata"):
        histogram = analyzed_rings[dataset]["n_nodes"]
        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob / np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        # entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        # print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs


class EGNN_dynamics(nn.Module):
    def __init__(
        self,
        in_node_nf,
        context_node_nf,
        n_dims,
        hidden_nf=64,
        device="cpu",
        act_fn=torch.nn.SiLU(),
        n_layers=4,
        attention=False,
        condition_time=True,
        tanh=False,
        mode="egnn_dynamics",
        norm_constant=0,
        inv_sublayers=2,
        sin_embedding=False,
        normalization_factor=100,
        aggregation_method="sum",
    ):
        super().__init__()
        self.mode = mode
        if mode == "egnn_dynamics":
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf,
                in_edge_nf=1,
                hidden_nf=hidden_nf,
                device=device,
                act_fn=act_fn,
                n_layers=n_layers,
                attention=attention,
                tanh=tanh,
                norm_constant=norm_constant,
                inv_sublayers=inv_sublayers,
                sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
            )
            self.in_node_nf = in_node_nf
        elif mode == "gnn_dynamics":
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3,
                in_edge_nf=0,
                hidden_nf=hidden_nf,
                out_node_nf=3 + in_node_nf,
                device=device,
                act_fn=act_fn,
                n_layers=n_layers,
                attention=attention,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
            )

        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)

        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs * n_nodes, 1)
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask
        x = xh[:, 0 : self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs * n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims :].clone()

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)

        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs * n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        if self.mode == "egnn_dynamics":
            h_final, x_final = self.egnn(
                h, x, edges, node_mask=node_mask, edge_mask=edge_mask
            )
            vel = (
                x_final - x
            ) * node_mask  # This masking operation is redundant but just in case
        elif self.mode == "gnn_dynamics":
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        if context is not None:
            # Slice off context size:
            h_final = h_final[:, : -self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print("Warning: detected nan, resetting EGNN output to zero.")
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if h_dims == 0:
            return vel
        else:
            h_final = h_final.view(bs, n_nodes, -1)
            return torch.cat([vel, h_final], dim=2)

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [
                    torch.LongTensor(rows).to(device),
                    torch.LongTensor(cols).to(device),
                ]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)


def get_model(args, dataloader_train):
    if args.conditioning:
        print(f"Conditioning on {args.target_features}")
        args.context_node_nf = dataloader_train.dataset.num_targets
    else:
        args.context_node_nf = 0
    prop_dist = DistributionProperty(args, dataloader_train)

    in_node_nf = dataloader_train.dataset.num_node_features
    nodes_dist = DistributionRings(getattr(args, "dataset", "cata"))

    # prop_dist = None
    # if len(args.conditioning) > 0:
    #     prop_dist = DistributionProperty(dataloader_train, args.conditioning)

    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print("Warning: dynamics model is _not_ conditioned on time.")
        dynamics_in_node_nf = in_node_nf

    net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf,
        context_node_nf=args.context_node_nf,
        n_dims=3,
        device=args.device,
        hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(),
        n_layers=args.n_layers,
        attention=args.attention,
        tanh=args.tanh,
        mode=args.model,
        norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers,
        sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor,
        aggregation_method=args.aggregation_method,
        coords_range=args.coords_range,
    )

    model = EnVariationalDiffusion(
        dynamics=net_dynamics,
        in_node_nf=in_node_nf,
        n_dims=3,
        timesteps=args.diffusion_steps,
        noise_schedule=args.diffusion_noise_schedule,
        noise_precision=args.diffusion_noise_precision,
        loss_type=args.diffusion_loss_type,
        norm_values=args.normalize_factors,
        include_charges=False,
        device=args.device,
    )

    if args.dp:  # and torch.cuda.device_count() > 1:
        print(f"Training using {torch.cuda.device_count()} GPUs")
        model = MyDataParallel(model)
    if args.restore is not None:
        model_state_dict = torch.load(args.exp_dir + "/model.pt")
        model.load_state_dict(model_state_dict)

    return model, nodes_dist, prop_dist


class DistributionProperty:
    def __init__(self, args, dataloader, num_bins=1000, normalizer=None):
        self.num_bins = num_bins
        self.distributions = {}
        self.properties = dataloader.dataset.target_features
        self.mean = dataloader.dataset.mean
        self.std = dataloader.dataset.std
        self.args = args
        for i, prop in enumerate(self.properties):
            self.distributions[prop] = {}
            data = torch.Tensor(dataloader.dataset.df[prop])
            if dataloader.dataset.normalize:
                data = (data - self.mean[i]) / self.std[i]
            self._create_prob_dist(
                torch.Tensor(dataloader.dataset.df["n_rings"]),
                data,
                self.distributions[prop],
            )

    def _create_prob_dist(self, nodes_arr, values, distribution):
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {"probs": probs, "params": params}

    def _create_prob_given_nodes(self, values):
        n_bins = self.num_bins  # min(self.num_bins, len(values))
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + 1e-12
        histogram = torch.zeros(n_bins)
        for val in values:
            i = int((val - prop_min) / prop_range * n_bins)
            # Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins-1)
            # We move it to bin int(n_bind-1 if tat happens)
            if i == n_bins:
                i = n_bins - 1
            histogram[i] += 1
        probs = histogram / torch.sum(histogram)
        probs = Categorical(probs)
        params = [prop_min, prop_max]
        return probs, params

    def sample(self, n_nodes=19):
        vals = []
        for prop in self.properties:
            dist = self.distributions[prop][n_nodes]
            idx = dist["probs"].sample((1,))
            val = self._idx2value(idx, dist["params"], len(dist["probs"].probs))
            vals.append(val)
        vals = torch.cat(vals)
        return vals

    def sample_batch(self, nodesxsample):
        vals = []
        for n_nodes in nodesxsample:
            vals.append(self.sample(int(n_nodes)).unsqueeze(0))
        vals = torch.cat(vals, dim=0)
        return vals

    def sample_df(self, nodesxsample, split="test"):
        df = getattr(self.args, f"df_{split}")
        vals = []
        for n_nodes in nodesxsample:
            val = df[df.n_rings == n_nodes.item()].sample(1)[self.properties].values
            vals.append(torch.Tensor(val))
        vals = torch.cat(vals, dim=0)
        return self.normalize(vals)

    def _idx2value(self, idx, params, n_bins):
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = torch.rand(1) * (right - left) + left
        return val

    def unnormalize(self, val):
        val = val * self.std.to(val.device) + self.mean.to(val.device)
        return val

    def normalize(self, val):
        val = (val - self.mean.to(val.device)) / self.std.to(val.device)
        return val


class EmpiricalDistributionProperty:
    def __init__(self, args, dataloader):
        self.distributions = {}
        self.properties = dataloader.dataset.target_features
        self.mean = dataloader.dataset.mean
        self.std = dataloader.dataset.std
        self.args = args
        self.data = torch.Tensor(dataloader.dataset.df[self.properties].values)

    def sample(self):
        return self.sample_batch(1)

    def sample_batch(self, nodesxsample):
        return self.normalize(
            self.data[torch.randperm(self.data.shape[0])[: len(nodesxsample)]]
        )

    def unnormalize(self, val):
        val = val * self.std.to(val.device) + self.mean.to(val.device)
        return val

    def normalize(self, val):
        val = (val - self.mean.to(val.device)) / self.std.to(val.device)
        return val
