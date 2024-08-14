import jax.numpy as jnp

from jax import vmap
from functools import partial
from fast_structure_learning import *


class RGM:

    def __init__(self, tile_diameter=16, max_levels=8, n_bins=9, sv_thr=1.0 / 32):
        self.tile_diameter = tile_diameter
        self.max_levels = max_levels
        self.n_bins = n_bins
        self.sv_thr = sv_thr

    def fit(self, observations, actions):
        self.image_shape = observations.shape[-3:]
        (
            observation_one_hots,
            self.locations_matrix,
            self.group_indices,
            self.sv_discrete_axis,
            self.V_per_patch,
        ), self.patch_indices = map_rgb_2_discrete(
            observations, tile_diameter=self.tile_diameter, n_bins=self.n_bins, sv_thr=self.sv_thr
        )
        action_one_hots, self.action_bins = map_action_2_discrete(actions, n_bins=self.n_bins)
        self.num_controls = action_one_hots.shape[0]

        # prepend action one_hots to first group
        self.group_indices = [0] * self.num_controls + self.group_indices
        one_hots = jnp.concatenate((action_one_hots, observation_one_hots), axis=0)
        self.agents, RG, LB = spm_mb_structure_learning(
            one_hots, self.locations_matrix, self.num_controls, max_levels=self.max_levels
        )

    def infer_states(self, observations, actions, priors=None):
        (observation_one_hots, locations_matrix, group_indices, sv_discrete_axis, V_per_patch), patch_indices = (
            map_rgb_2_discrete(
                observations,
                tile_diameter=self.tile_diameter,
                n_bins=self.n_bins,
                V_per_patch=self.V_per_patch,
                sv_discrete_axis=self.sv_discrete_axis,
            )
        )
        action_one_hots, _ = map_action_2_discrete(actions, n_bins=self.n_bins)
        one_hots = jnp.concatenate((action_one_hots, observation_one_hots), axis=0)
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
        map_discrete_2_rgb_fn = partial(
            map_discrete_2_rgb,
            locations_matrix=self.locations_matrix,
            group_indices=self.group_indices[self.num_controls :],
            sv_discrete_axis=self.sv_discrete_axis,
            V_per_patch=self.V_per_patch,
            patch_indices=self.patch_indices,
            image_shape=self.image_shape,
        )

        action_one_hots = observations[0][: self.num_controls]
        actions = map_discrete_2_action(action_one_hots, self.action_bins)

        observation_one_hots = observations[0][self.num_controls :]
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

    def act(self, observations):
        # add latest observations to a sliding window and infer/predict
        pass
