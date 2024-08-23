import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            learning_rate,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO: implement the forward pass of the critic network
        output = self.network(obs)
        return output

        

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        
        # In case of data conversion
        if isinstance(obs, list):
            obs = np.array(obs)
        if isinstance(q_values, list):
            q_values= np.array(q_values)

        obs = ptu.from_numpy(obs)
        q_values = ptu.from_numpy(q_values)

        # TODO: update the critic using the observations and q_values
        ## What is the loss of Critic?
        ## Mostly, it is L2 regression.
        self.optimizer.zero_grad()

        loss_fn = nn.MSELoss()

        pred_q_values = self(obs)

        ## FIXME) SM: This would be generally L2-regression loss
        loss = loss_fn(
            pred_q_values, 
            q_values
        )

        loss.backward()
        self.optimizer.step()

        return {
            "Baseline Loss": ptu.to_numpy(loss),
        }