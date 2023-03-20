import sys
from time import time
import pandas as pd
from rdkit import Chem
import pickle
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from edm.equivariant_diffusion.en_diffusion import (
    PredefinedNoiseSchedule,
    softplus,
    expm1,
)


def sigma(g):
    return torch.sqrt(torch.sigmoid(torch.Tensor(g)))


def sigma_and_alpha_t_given_s(
    gamma_t: torch.Tensor,
    gamma_s: torch.Tensor,
):
    """
    Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

    These are defined as:
        alpha t given s = alpha t / alpha s,
        sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
    """
    sigma2_t_given_s = -expm1(softplus(gamma_s) - softplus(gamma_t))

    # alpha_t_given_s = alpha_t / alpha_s
    log_alpha2_t = F.logsigmoid(-gamma_t)
    log_alpha2_s = F.logsigmoid(-gamma_s)
    log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

    alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
    alpha_t_given_s = alpha_t_given_s
    sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

    return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s


gamma = PredefinedNoiseSchedule("polynomial_2", 1000, 1e-5).show_schedule(1000)
gamma = torch.Tensor(gamma)
w1, w2 = [], []
range_w = torch.linspace(0, 998, 100).long()
for i in range_w:
    i = i.item()
    s = i / 1000
    t = (i + 1) / 1000
    gamma_s = gamma[i]
    gamma_t = gamma[i + 1]

    (
        sigma2_t_given_s,
        sigma_t_given_s,
        alpha_t_given_s,
    ) = sigma_and_alpha_t_given_s(gamma_t, gamma_s)

    sigma_s = sigma(gamma_s)
    sigma_t = sigma(gamma_t)
    sigma1 = sigma_t_given_s * sigma_s / sigma_t

    w1.append(sigma1)
    w2.append(sigma2_t_given_s / alpha_t_given_s)

cond_pred_error_x = torch.linspace(0, 1000, 11)
cond_pred_error_y = [
    0.0068,
    0.0075,
    0.0089,
    0.0135,
    0.022,
    0.045,
    0.0792,
    0.1132,
    0.1341,
    0.1432,
    0.1525,
]


# plt.plot(range_w, w1)
# plt.plot(range_w, w2)
# plt.legend(["Ours (Sigma)", "EEGSDE"])
# plt.show()

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
l1 = ax1.plot(range_w, w1, c="r")
l2 = ax2.plot(cond_pred_error_x, cond_pred_error_y, c="b")

ax1.set_xlabel("Timestamp")
ax1.set_ylabel("Noise level - noise std")
ax2.set_ylabel("Conditional predictor error - MAE")
plt.legend(l1 + l2, ["Noise level", "Conditional predictor error"])
plt.gca().invert_xaxis()
plt.savefig("prediction.pdf", dpi=300, bbox_inches="tight")
plt.show()
