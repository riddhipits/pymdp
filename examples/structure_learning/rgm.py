import jax.numpy as jnp

from jax import vmap
from functools import partial
from pymdp.jax.agent import Agent
from fast_structure_learning import *


class RGM:

    def __init__(self, tile_diameter=16, max_levels=8, n_bins=9, sv_thr=1.0 / 32):
        self.tile_diameter = tile_diameter
        self.max_levels = max_levels
        self.n_bins = n_bins
        self.sv_thr = sv_thr

        self.V_per_patch = None
        self.sv_discrete_axis = None

    def fit(self, observations, actions):
        self.image_shape = jnp.asarray(observations.shape[-3:])
        one_hots = self.to_one_hot(observations, actions)
        # TODO get the number of timesteps for one_hot observations that yield one top-level qs
        self.observation_shape = jnp.asarray([one_hots.shape[0], 2, one_hots.shape[2]])
        self.agents, RG, LB = spm_mb_structure_learning(
            one_hots, self.locations_matrix, self.num_controls, max_levels=self.max_levels
        )

    def save(self, filename):
        # save model fit to .npz file
        data = {}
        # info from SVD to discretize RGB
        for i in range(len(self.V_per_patch)):
            data[f"V_per_patch_{i}"] = self.V_per_patch[i]
        for i in range(len(self.sv_discrete_axis)):
            data[f"sv_discrete_axis_{i}"] = self.sv_discrete_axis[i]
        data["locations_matrix"] = self.locations_matrix
        for i in range(len(self.patch_indices)):
            data[f"patch_indices_{i}"] = self.patch_indices[i]
        data["group_indices"] = jnp.asarray(self.group_indices)

        # info to discretize actions
        data["action_bins"] = self.action_bins

        # A and B tensors for agents
        for i in range(len(self.agents)):
            for j in range(len(self.agents[i].A)):
                data[f"A_{i}_{j}"] = self.agents[i].A[j]
                data[f"A_dependencies_{i}_{j}"] = jnp.asarray(self.agents[i].A_dependencies[j])
        for i in range(len(self.agents)):
            for j in range(len(self.agents[i].B)):
                data[f"B_{i}_{j}"] = self.agents[i].B[j]
                data[f"B_dependencies_{i}_{j}"] = jnp.asarray(self.agents[i].B_dependencies[j])

        # generic shapes and controls
        data["observation_shape"] = self.observation_shape
        data["image_shape"] = self.image_shape
        data["num_controls"] = jnp.array([self.num_controls])
        jnp.savez(filename, **data)

    def load(self, filename):
        # load model fit from .npz file
        data = jnp.load(filename)

        self.locations_matrix = jnp.asarray(data["locations_matrix"])
        self.patch_indices = []
        i = 0
        while f"patch_indices_{i}" in data:
            self.patch_indices.append(jnp.asarray(data[f"patch_indices_{i}"]))
            i += 1
        self.group_indices = data["group_indices"].tolist()
        self.V_per_patch = []
        i = 0
        while f"V_per_patch_{i}" in data:
            self.V_per_patch.append(jnp.asarray(data[f"V_per_patch_{i}"]))
            i += 1
        self.sv_discrete_axis = []
        i = 0
        while f"sv_discrete_axis_{i}" in data:
            self.sv_discrete_axis.append(jnp.asarray(data[f"sv_discrete_axis_{i}"]))
            i += 1

        self.action_bins = jnp.asarray(data["action_bins"])

        agents = []
        for i in range(self.max_levels):
            if f"A_{i}_0" not in data:
                break

            A = []
            A_dependencies = []
            j = 0
            while f"A_{i}_{j}" in data:
                A.append(data[f"A_{i}_{j}"])
                A_dependencies.append(data[f"A_dependencies_{i}_{j}"].tolist())
                j += 1

            B = []
            B_dependencies = []
            j = 0
            while f"B_{i}_{j}" in data:
                B.append(data[f"B_{i}_{j}"])
                B_dependencies.append(data[f"B_dependencies_{i}_{j}"].tolist())
                j += 1

            agents.append(Agent(A, B, A_dependencies=A_dependencies, B_dependencies=B_dependencies, apply_batch=False))
        self.agents = agents

        self.observation_shape = data["observation_shape"]
        self.image_shape = data["image_shape"]
        self.num_controls = data["num_controls"][0]

    def to_one_hot(self, observations, actions=None, mask_indices=None):
        (
            observation_one_hots,
            self.locations_matrix,
            self.group_indices,
            self.sv_discrete_axis,
            self.V_per_patch,
        ), self.patch_indices = map_rgb_2_discrete(
            observations,
            tile_diameter=self.tile_diameter,
            n_bins=self.n_bins,
            sv_thr=self.sv_thr,
            V_per_patch=self.V_per_patch,
            sv_discrete_axis=self.sv_discrete_axis,
        )
        if actions is None:
            action_one_hots = jnp.ones([self.num_controls, observation_one_hots.shape[1], self.n_bins]) / self.n_bins
        else:
            action_one_hots, self.action_bins = map_action_2_discrete(actions, n_bins=self.n_bins)
            self.num_controls = action_one_hots.shape[0]

        mask_val = 1.0 / observation_one_hots.shape[-1]
        if mask_indices is not None:
            for idx in range(len(self.group_indices)):
                if self.group_indices[idx] in mask_indices:
                    observation_one_hots = observation_one_hots.at[idx, ...].set(mask_val)

        # prepend action one_hots to first group
        self.group_indices = [0] * self.num_controls + self.group_indices
        one_hots = jnp.concatenate((action_one_hots, observation_one_hots), axis=0)
        return one_hots

    def infer_states(self, observations, actions=None, priors=None, one_hot_obs=False):
        if not one_hot_obs:
            one_hots = self.to_one_hot(observations, actions)
        else:
            one_hots = observations
        qs = infer(self.agents, one_hots, priors)
        return qs

    def infer_empirical_prior(self, path, qs):
        o, prior = predict(self.agents, qs, path, num_steps=1)

        # only keep the predicted part
        for i in range(len(prior)):
            prior[i] = prior[i][:, prior[i].shape[1] // 2 :, ...]

        return prior

    def reconstruct(self, qs):
        observations, _ = predict(self.agents, qs, None, num_steps=0)
        return self.discrete_2_rgb_action(observations[0])

    def discrete_2_rgb_action(self, one_hots):
        map_discrete_2_rgb_fn = partial(
            map_discrete_2_rgb,
            locations_matrix=self.locations_matrix,
            group_indices=self.group_indices[self.num_controls :],
            sv_discrete_axis=self.sv_discrete_axis,
            V_per_patch=self.V_per_patch,
            patch_indices=self.patch_indices,
            image_shape=self.image_shape,
        )

        action_one_hots = one_hots[: self.num_controls]
        actions = map_discrete_2_action(action_one_hots, self.action_bins)

        observation_one_hots = one_hots[self.num_controls :]
        imgs = vmap(map_discrete_2_rgb_fn, in_axes=1, out_axes=0)(observation_one_hots)
        imgs = imgs.reshape((imgs.shape[0] * imgs.shape[1], imgs.shape[-3], imgs.shape[-2], imgs.shape[-1]))

        # convert to plot-able format
        imgs = jnp.transpose(imgs, (0, 2, 3, 1))
        imgs /= 255
        imgs = jnp.clip(imgs, 0, 1)
        imgs = (255 * imgs).astype(jnp.uint8)
        return imgs, actions

    def infer_policies(self, qs):
        pass

    def sample_action(self, q_pi):
        pass


class RGMAgent:

    def __init__(self, rgm: RGM):
        self.rgm = rgm

        self.priors = None
        self.observations = jnp.ones(jnp.asarray(rgm.observation_shape)) / rgm.observation_shape[-1]
        self.t = 0

        self.posterior = jnp.ones([1, rgm.agents[-1].A[0].shape[-1]])
        self.posterior = [self.posterior / jnp.sum(self.posterior)]
        self.window = []

    def act(self, obs):
        # add latest observations to a sliding window and infer/predict
        self.window.append(obs["pixels"])
        if len(self.window) == 2:
            # we got two new images, get one-hots
            stacked = jnp.stack(self.window)
            obs = self.rgm.to_one_hot(stacked)
            # TODO set idx = 0 (t = 1) or  idx=1 t = 3
            if self.t == 1:
                self.observations = self.observations.at[:, :1, :].set(obs)
            elif self.t == 3:
                self.observations = self.observations.at[:, 1:, :].set(obs)
            self.posterior = self.rgm.infer_states(self.observations, priors=self.priors, one_hot_obs=True)
            self.window = []

            # TODO update posterior if we have enough data?
        imgs, actions = self.rgm.reconstruct(self.posterior)

        a = actions[self.t]
        img = imgs[self.t]

        # TODO should we clamp action observations to selected actions?

        self.t += 1

        if self.t == 4:
            # TODO don't hardcode t == x
            # forward 1 tick at the higest level
            self.t = 0
            # TODO "infer" the path at the highest level
            self.priors = self.rgm.infer_empirical_prior(jnp.array([[0]]), self.posterior)
            self.posterior = [self.priors[len(self.rgm.agents) - 1][0]]
            self.observations = jnp.ones_like(self.observations) / self.observations.shape[-1]
            self.window = []

        return a, img
