from typing import Optional, Sequence
import numpy as np
import torch

from cs285.networks.policies import MLPPolicyPG
from cs285.networks.critics import ValueCritic
from cs285.infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # TODO: flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.

        
        """
        SM: don't reshape -> just concat it
        """
        obs = np.concatenate(obs, axis = 0)
        actions = np.concatenate(actions, axis = 0)
        rewards = np.concatenate(rewards, axis = 0)
        terminals = np.concatenate(terminals, axis = 0)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        print(f"""Check shape:
              obs -> {obs.shape}
              actions -> {actions.shape}
              rewards -> {rewards.shape}
              advs -> {advantages.shape}
              """)


        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # TODO: update the PG actor/policy network once using the advantages
        # FIXME) I think the network should be done.
        info: dict = {} 

        info['obs'] = obs 
        info['actions'] = actions 
        info['advantages'] = advantages 
        self.actor.update(**info)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # TODO: perform `self.baseline_gradient_steps` updates to the critic/baseline network
            critic_info: dict = {}

            for _ in range(self.baseline_gradient_steps):
                critic_info['obs'] = obs
                critic_info['q_values'] = q_values 
                self.critic.update(**critic_info)

            ## FIXME) After update, let it go sangmo.
            info.update(critic_info)

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            # TODO: use the helper function self._discounted_return to calculate the Q-values
            q_values = self._discounted_return(rewards)
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # TODO: use the helper function self._discounted_reward_to_go to calculate the Q-values
            q_values =self._discounted_reward_to_go(rewards)

        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """

        print("Check L147 for intentional fix")
        q_values = np.array(q_values) 
        print(f"""Shape check for _estimate_advantage(L147 of pg_agent.py)
            obs ->{obs.shape}
            rewards ->{rewards.shape}
            q_values ->{q_values.shape}
            terminals ->{terminals.shape}
            """)


        if self.critic is None:
            # TODO: if no baseline, then what are the advantages?
            # FIXME: SM=> If no baseline, the advantage could be defined as the mean of Q-function
            mean_baseline_arr = np.mean(q_values)
            advantages = q_values - mean_baseline_arr
        else:
            # TODO: run the critic and use it as a baseline
            # FIXME: critic network config is already defined in self.critic(check __init__.py)
            # Thus, simply pass obs to the network

            # Convert to tensor first 
            obs = ptu.from_numpy(obs)
            values = self.critic(obs)
            values = ptu.to_numpy(values).flatten()

            print("Check shape again:", values.shape)

            debug_string = ("Sangmo, the observation may be various. You may pass Value network serveral times to get value per timestep"
                            "Checkout carefully!")

            print(debug_string)
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                # TODO: if using a baseline, but not GAE, what are the advantages?
                # FIXME: We already have values(only current timestep though), q_values, and rewards.
                # Simply do Q - Value!
                advantages = q_values - values 
            else:
                # TODO: implement GAE
                batch_size = obs.shape[0]

                # HINT: append a dummy T+1 value for simpler recursive calculation
                ## FIXME: 여기서의 value는 서로 다른 trajectory에서 온 것일 수도 있지 않나?
                ## 어 아마 그럴거고, terminals 같은걸로 loop 돌아가면서 N-return 만들어야겠지.
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                # (i+1) th index is set for recursion 
                for i in reversed(range(batch_size)):
                    next_value = values[i+1]
                    cur_value = values[i] #V(s_t)
                    cur_reward = rewards[i] # r(s_t, a_t)

                    trajectory_end = terminals[i]

                    # If end-of-trajectory: set cur_reward to 0
                    if trajectory_end:
                        cur_advantage = 0 
                        gae_advantage = cur_value 

                    else:
                        # TD-0 advantage
                        cur_advantage = cur_reward + self.gamma * next_value

                        # Thank you nice mathematcian. Recursion of GAE estimator saved me!!!
                        gae_advantage = cur_advantage + self.gamma * self.gae_lambda * gae_advantage
                    
                    # Either way, set advantage to gae_advantage
                    advantages[i] = gae_advantage

    
                # remove dummy advantage
                advantages = advantages[:-1]

        # TODO: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            mean = np.mean(advantages)
            std = np.std(advantages)
            advantages = (advantages - mean)/ std
            breakpoint()
            pass

        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """

        # SM: 하나씩 거꾸로 돌아가면서 추가됌. => 상모야... rewards 받은 timestep 별로 reward가 있다고 가정해야지.
        # TODO) Add reward in cumulative manner.

        print("Check reward shape L218:", len(rewards))

        """
        SM) 
        When you check the specific input of this function, it seems that the rewards array is in-fact 2D.
        That is, rewards has the shape of (B, ???) array.

        Batch means each trajectory.
        ??? means length of totaj trajectory.        
        """
        

        output_list = []
        for per_trajectory_reward in rewards:
            cumul_reward = 0
            trajectory_q_value_list = []
            for t, reward in enumerate(reversed(per_trajectory_reward)):
                cumul_reward += (self.gamma ** t) * reward
            # Since every index should contain the total sum, replicate the cumulative sum for each entry
                trajectory_q_value_list.append(cumul_reward)
            # Q-value was added 
            trajectory_q_value_list.reverse()
            # Extend to original list
            output_list.extend(trajectory_q_value_list)
        
        return output_list


    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """

        if self.critic is None: 
            target_timestep = 0        
        else:
            target_timestep = self.baseline_gradient_steps
        total_reward_to_go_list = []
        cumul_reward = 0

        total_reward_to_go_list = []

        for per_trajectory_reward in rewards:
            cumul_reward = 0
            reward_to_go_q_list = []
            for t, reward in enumerate(reversed(per_trajectory_reward)):
                # Add until reward-to-go point
                if t < len(per_trajectory_reward) - target_timestep :
                    cumul_reward += (self.gamma ** t) * reward
                reward_to_go_q_list.append(cumul_reward)
            # Q-value was added 
            reward_to_go_q_list.reverse()
            # Extend to original list
            total_reward_to_go_list.extend(reward_to_go_q_list)

        return total_reward_to_go_list
