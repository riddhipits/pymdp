import cv2
import numpy as onp
from jax import numpy as jnp
from jax import vmap
from jax import nn
from jax import tree_util as jtu
from jaxtyping import Array
from typing import Tuple, List
import itertools
import matplotlib.pyplot as plt
import imageio

from pymdp.jax.agent import Agent
from pymdp.jax.control import compute_expected_obs, compute_expected_state
from pymdp.jax.maths import factor_dot
from pymdp.jax.algos import run_factorized_fpi
from pymdp.jax.learning import update_state_transition_dirichlet

from functools import partial

from math import prod


def read_frames_from_npz(file_path: str, num_frames: int = 32, rollout: int = 0):
    """read frames from a npz file from atari expert trajectories"""
    # shape is [num_rollouts, num_frames, 1, height, width, channels]
    res = onp.load(file_path)
    frames = res["arr_0"][rollout, 0:num_frames, 0, ...]
    return frames


def read_frames_from_mp4(file_path: str, num_frames: int = 32, size: tuple[int] = (128, 128)):
    """ " read frames from an mp4 file"""
    cap = cv2.VideoCapture(file_path)

    width, height = size
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    x_center = video_width // 2
    y_center = video_height // 2
    # x_start = max(0, x_center - width // 2)
    # y_start = max(0, y_center - height // 2)
    x_start = max(0, x_center - width)
    y_start = max(0, y_center - height)

    frame_indices = jnp.linspace(0, total_frames - 1, num_frames, dtype=int)
    frame_indices = jnp.concatenate((frame_indices, frame_indices), axis=0)
    frames = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx.item()))

        ret, frame = cap.read()
        if not ret:
            break

        # cropped_frame = frame[y_start:y_start+height, x_start:x_start+width]
        cropped_frame = frame[y_start : y_start + 2 * height, x_start : x_start + 2 * width]
        resized_frame = cv2.resize(cropped_frame, size)
        frames.append(resized_frame)

    cap.release()
    return jnp.array(frames)


def map_rgb_2_discrete(
    image_data: Array,
    tile_diameter=32,
    n_bins=9,
    max_n_modes=32,
    sv_thr=(1.0 / 32.0),
    t_resampling=2,
    V_per_patch=None,
    sv_discrete_axis=None,
):
    """Re-implementation of `spm_rgb2O.m` in Python
    Maps an RGB image format to discrete outcomes

    Args:
    image_data: Array
        The image data to be mapped to discrete outcomes. Shape (num_frames, width, height, channels)
    tile_diameter: int
        Diameter of the tiles (`nd` in the original code)
    n_bins: int
        Number of variates (`nb` in the original code)
    max_n_modes: int
        Maximum number of modes (`nm` in the original code)
    sv_thr: int
        Threshold for singular values (`su` in the original code)
    t_resampling: int
        Threshold for temporal resampling (`R` in the original code)
    V_per_patch: List[Array]
        List of eigenvectors per patch from a previous SVD
    sv_discrete_axis: List[Array]
        List of quantization bins for singular variates from a previous SVD
    """
    # ensure number of bins is odd (for symmetry)
    n_bins = int(2 * jnp.trunc(n_bins / 2) + 1)

    n_frames, width, height, n_channels = image_data.shape
    T = int(t_resampling * jnp.trunc(n_frames / t_resampling))  # length of time partition

    # transpose to [T x C x W x H]
    image_data = jnp.transpose(
        image_data[:T, ...], (0, 3, 1, 2)
    )  # truncate the time series and transpose the axes to the right place

    # concat each t_resampling frames
    image_data = image_data.reshape((T // t_resampling, -1, width, height))

    # shape of the data excluding the time dimension ((t_resampling * C) x W x H)
    shape_no_time = image_data.shape[1:]

    patch_indices, patch_centroids, patch_weights = spm_tile(
        width=shape_no_time[1], height=shape_no_time[2], n_copies=shape_no_time[0], tile_diameter=tile_diameter
    )

    return (
        patch_svd(
            image_data,
            patch_indices,
            patch_centroids,
            patch_weights,
            sv_thr=sv_thr,
            max_n_modes=max_n_modes,
            n_bins=n_bins,
            V_per_patch=V_per_patch,
            sv_discrete_axis=sv_discrete_axis,
        ),
        patch_indices,
    )


def patch_svd(
    image_data: Array,
    patch_indices: List[Array],
    patch_centroids,
    patch_weights: List[Array],
    sv_thr: float = 1e-6,
    max_n_modes: int = 32,
    n_bins: int = 9,
    V_per_patch=None,
    sv_discrete_axis=None,
):
    """
    image_data: [time, channel, width, height]
    patch_indices: [[indicies_for_patch] for num_patches]
    patch_weights: [[indices_for_patch, num_patches] for num_patches]
    """
    n_frames, channels_x_duplicates, width, height = image_data.shape
    o_idx = 0
    observations = []
    locations_matrix = []
    group_indices = []

    # if V_per_patch and sv_discrete_axis are not provided, we need to do SVD
    do_svd = False
    if V_per_patch is None or sv_discrete_axis is None:
        sv_discrete_axis = []
        V_per_patch = [None] * len(patch_indices)
        do_svd = True

    # iterate over all patches
    for g_i, patch_g_indices in enumerate(patch_indices):

        # get the pixels of each patch, weighted by their distance to the centroid
        Y = (
            image_data.reshape(n_frames, channels_x_duplicates * width * height)[:, patch_g_indices]
            * patch_weights[g_i]
        )

        # new_image = jnp.zeros(channels_x_duplicates*width*height)
        # new_image = new_image.at[patch_g_indices].set(patch_weights[g_i])
        # new_image = new_image.reshape(channels_x_duplicates, width, height)
        # plt.imshow(new_image[0])
        # plt.title(f'Patch {g_i}')
        # plt.show()

        if not do_svd:
            V = V_per_patch[g_i]
            u = Y @ V

            num_modalities = u.shape[1]
            for m in range(num_modalities):
                observations.append([])
                for t in range(u.shape[0]):
                    min_indices = jnp.argmin(jnp.abs(u[t, m] - sv_discrete_axis[o_idx]))
                    observations[o_idx].append(nn.one_hot(min_indices, n_bins))

                locations_matrix.append(patch_centroids[g_i, :])
                group_indices.append(g_i)
                o_idx += 1
        else:
            # (n_frames x n_frames), (n_frames,), (n_frames x n_frames)
            U, svals, V = jnp.linalg.svd(Y @ Y.T, full_matrices=True)

            # plt.imshow(Y@Y.T)
            # plt.title(f'Patch {g_i}')
            # plt.show()

            normalized_svals = svals * (len(svals) / svals.sum())
            topK_svals = normalized_svals > sv_thr  # equivalent of `j` in spm_svd.m
            topK_s_vectors = U[:, topK_svals]

            projections = Y.T @ topK_s_vectors  # do equivalent of spm_en on this one
            projections_normed = projections / jnp.linalg.norm(projections, axis=0, keepdims=True)

            svals = jnp.sqrt(svals[topK_svals])

            num_modalities = min(len(svals), max_n_modes)

            if num_modalities > 0:
                V_per_patch[g_i] = projections_normed[:, :num_modalities]
                weighted_topk_s_vectors = topK_s_vectors[:, :num_modalities] * svals[:num_modalities]

            # generate (probability over discrete) outcomes
            for m in range(num_modalities):

                # discretise singular variates
                d = jnp.max(jnp.abs(weighted_topk_s_vectors[:, m]))

                # this determines the number of bins
                projection_bins = jnp.linspace(-d, d, n_bins)

                observations.append([])
                for t in range(n_frames):

                    # finds the index of of the projection at time t, for singular vector m, in the projection bins -- this will determine how it gets discretized
                    min_indices = jnp.argmin(jnp.absolute(weighted_topk_s_vectors[t, m] - projection_bins))

                    # observations are a one-hot vector reflecting the quantization of each singular variate into one of the projection bins
                    observations[o_idx].append(nn.one_hot(min_indices, n_bins))

                # record locations and group for this outcome
                locations_matrix.append(patch_centroids[g_i, :])
                group_indices.append(g_i)
                sv_discrete_axis.append(projection_bins)
                o_idx += 1

    locations_matrix = jnp.stack(locations_matrix)
    observations = jnp.asarray(observations)

    return observations, locations_matrix, group_indices, sv_discrete_axis, V_per_patch


def map_discrete_2_rgb(
    observations,
    locations_matrix,
    group_indices,
    sv_discrete_axis,
    V_per_patch,
    patch_indices,
    image_shape,
    t_resampling=2,
):
    n_groups = len(patch_indices)

    # image_shape given as [W H C]
    shape = [t_resampling, image_shape[-1], image_shape[-3], image_shape[-2]]

    recons_image = jnp.zeros(prod(shape))

    for group_idx in range(n_groups):
        modality_idx_in_patch = [modality_idx for modality_idx, g_i in enumerate(group_indices) if g_i == group_idx]
        num_modalities_in_patch = len(modality_idx_in_patch)

        matched_bin_values = []
        for m in range(num_modalities_in_patch):
            m_idx = modality_idx_in_patch[m]
            matched_bin_values.append(sv_discrete_axis[m_idx].dot(observations[m_idx]))

        matched_bin_values = jnp.array(matched_bin_values)
        if len(matched_bin_values) > 0:
            recons_image = recons_image.at[patch_indices[group_idx]].set(
                recons_image[patch_indices[group_idx]] + V_per_patch[group_idx].dot(matched_bin_values)
            )

    recons_image = recons_image.reshape(shape)
    return recons_image


def map_action_2_discrete(actions, n_bins=9, min_val=-1, max_val=1):
    """
    Maps continuous actions to discrete outcomes
    Args:
        actions: Array
            The actions to be mapped to discrete outcomes. Shape (num_frames, num_actions)
        n_bins: int
            Number of bins to quantize the actions into
    """

    action_bins = jnp.linspace(min_val, max_val, n_bins)
    min_indices = jnp.argmin(jnp.absolute(actions[:, :, jnp.newaxis] - action_bins), axis=-1)
    one_hots = nn.one_hot(min_indices, n_bins)

    reshaped = one_hots.reshape(-1, one_hots.shape[1] * 2, n_bins)
    transposed = jnp.transpose(reshaped, (1, 0, 2))
    return transposed, action_bins


def map_discrete_2_action(action_one_hots, action_bins):
    """
    Maps discrete outcomes to continuous actions
    Args:
        action_one_hots: Array
            The one-hot encoded actions. Shape (num_frames, num_actions, num_bins)
        action_bins: Array
            The bins to quantize the actions into. Shape (num_bins,)
    """

    transposed = jnp.transpose(action_one_hots, (1, 0, 2))
    actions = jnp.sum(transposed * action_bins, axis=-1)
    actions = actions.reshape(-1, actions.shape[1] // 2)
    return actions


def spm_dir_norm(a):
    """
    Normalisation of a (Dirichlet) conditional probability matrix
    Args:
        A: (Dirichlet) parameters of a conditional probability matrix
    Returns:
        A: normalised conditional probability matrix
    """
    a0 = jnp.sum(a, axis=0)
    i = a0 > 0
    a = jnp.where(i, a / a0, a)
    a = a.at[:, ~i].set(1 / a.shape[0])
    return a


def spm_tile(width: int, height: int, n_copies: int, tile_diameter: int = 32):
    """
    Grouping into a partition of non-overlapping outcome tiles
    This routine identifies overlapping groups of pixels, returning their
    mean locations and a cell array of weights (based upon radial Gaussian
    basis functions) for each group. In other words, the grouping is based
    upon the location of pixels; in the spirit of a receptive field afforded
    by a sensory epithelium.Effectively, this leverages the conditional
    independencies that inherit from local interactions; of the kind found in
    metric spaces that preclude action at a distance.

    Args:
        L: list of indices
        width: width of the image
        height: height of the image
        tile_diameter: diameter of the tiles
    Returns:
        G: outcome indices
        M: (mean) outcome location
        H: outcome weights
    """

    def distance(x, y):
        return jnp.sqrt(((x - y) ** 2).sum())

    def flatten(l):
        return [item for sublist in l for item in sublist]

    # Centroid locations
    n_rows = int((width + 1) / tile_diameter)
    n_columns = int((height + 1) / tile_diameter)
    x = jnp.linspace(tile_diameter / 2 - 1, width - tile_diameter / 2, n_rows)
    y = jnp.linspace(tile_diameter / 2 - 1, height - tile_diameter / 2, n_columns)

    pixel_indices = n_copies * [jnp.array(jnp.meshgrid(jnp.arange(width), jnp.arange(height))).T.reshape(-1, 2)]
    pixel_indices = jnp.concatenate(pixel_indices, axis=0)

    h = [[[] for _ in range(n_columns)] for _ in range(n_rows)]
    g = [[None for _ in range(n_columns)] for _ in range(n_rows)]
    for i in range(n_rows):
        for j in range(n_columns):
            pos = jnp.array([x[i], y[j]])
            distance_evals = vmap(lambda x: distance(x, pos))(pixel_indices)

            ij = jnp.argwhere(distance_evals < (2 * tile_diameter)).squeeze()
            h[i][j] = jnp.exp(-jnp.square(distance_evals) / (2 * (tile_diameter / 2) ** 2))
            g[i][j] = ij

    G = flatten(g)
    h_flat = flatten(h)

    num_groups = n_rows * n_columns

    # weighting of groups
    h_matrix = jnp.stack(h_flat)  # [num_groups, n_pixels_per_group)
    h = spm_dir_norm(h_matrix)  # normalize across groups

    H_weights = [h[g_i, G[g_i]] for g_i in range(num_groups)]

    M = jnp.zeros((num_groups, 2))
    for g_i in range(num_groups):
        M = M.at[g_i, :].set(pixel_indices[G[g_i], :].mean(0))

    return G, M, H_weights


def spm_space(L: Array):
    """
    This function takes a set of modalities and their
    spatial coordinates and decimates over space into a compressed
    set of modalities, and assigns the previous modalities
    to the new set of modsalities.

    Args:
        L (Array): num_modalities x 2
    Returns:
        G (List[Array[int]]):
            outcome indices mapping new modalities indices to
            previous modality indices
    """

    # this is the second case (skipping if isvector(L))
    # locations
    Nl = L.shape[0]
    unique_locs = jnp.unique(L, axis=0)
    Ng = unique_locs.shape[0]
    Ng = jnp.ceil(jnp.sqrt(Ng / 4))
    if Ng == 1:
        G = jnp.arange(Nl)
        return [G]

    # decimate locations
    x = jnp.linspace(jnp.min(L[:, 0]), jnp.max(L[:, 0]), int(Ng))
    y = jnp.linspace(jnp.min(L[:, 1]), jnp.max(L[:, 1]), int(Ng))
    R = jnp.fliplr(jnp.array(jnp.meshgrid(x, y)).T.reshape(-1, 2))

    # nearest (reduced) location
    closest_loc = lambda loc: jnp.argmin(jnp.linalg.norm(R - loc, axis=1))
    g = vmap(closest_loc)(L)

    # grouping partition
    G = []

    # these two lines do the equivalent of u = unique(g, 'stable') in MATLAB
    _, unique_idx = jnp.unique(g, return_index=True)
    u = g[jnp.sort(unique_idx)]
    for i in range(len(u)):
        G.append(jnp.argwhere(g == u[i]).squeeze())

    return G


def spm_time(T, d):
    """
    Grouping into a partition of non-overlapping sequences
    Args:
    T (int): total number of the timesteps
    d (int): number timesteps per partition

    Returns:
    list: A list of partitions with non-overlapping sequences
    """
    t = []
    for i in range(T // d):
        t.append(jnp.arange(d) + (i * d))
    return t


def spm_unique(a):
    """
    Fast approximation by simply identifying unique locations in a
    multinomial statistical manifold, after discretising to probabilities of
    zero, half and one (using Matlab’s unique and fix operators).

    Args:
        a: array (n, x)
    Returns:
        indices of unique x'es
    """

    # Discretize to probabilities of zero, half, and one
    # 0 to 0.5 -> 0, 0.5 to 1 -> 1, 1 -> 2
    o_discretized = jnp.fix(2 * a)

    # Find unique rows -- this however needs to be changed to mimic the behavior of unique(o_discretized, 'stable')
    _, j = jnp.unique(o_discretized, return_inverse=True, axis=0)

    # suddenly j no longer has trailing 1 dimension?!
    if j.shape[-1] == 1:
        j = j.squeeze(axis=1)
    return j


def spm_structure_fast(observations, dt=2):
    """
    Args:
        observations (array): (num_modalities, num_steps, num_obs)
        dt (int)
    """

    # Find unique outputs per timestep
    num_modalities, num_steps, num_obs = observations.shape
    o = jnp.moveaxis(observations, 1, 0).reshape(num_steps, -1)
    j = spm_unique(o)

    # Likelihood tensors
    Ns = len(jnp.unique(j))  # number of latent causes

    a = num_modalities * [None]

    for m in range(num_modalities):
        a[m] = jnp.zeros((num_obs, Ns))
        for s in range(Ns):
            a[m] = (
                a[m].at[:, s].set(observations[m, j == s].mean(axis=0))
            )  # observations[m,j == s] will have shape (num_timesteps_that_match, num_bins)

    # Transition tensors
    if dt < 2:
        # no dynamics
        b = [jnp.eye(Ns)]
        return a, b

    # Assign unique transitions between states to paths
    b = jnp.zeros((Ns, Ns, 1))

    for t in range(len(j) - 1):
        if not jnp.any(b[j[t + 1], j[t], :]):
            # does this state have any transitions under any paths
            u = jnp.where(~jnp.any(b[:, j[t], :], axis=0))[0]
            if len(u) == 0:
                # Add new path if no empty paths found
                b = jnp.concatenate((b, jnp.zeros((Ns, Ns, 1))), axis=2)
                b = b.at[j[t + 1], j[t], -1].set(1)
            else:
                # Use first empty path
                b = b.at[j[t + 1], j[t], u].set(1)

    return a, [b]


def spm_mb_structure_learning(observations, locations_matrix, num_controls=0, dt: int = 2, max_levels: int = 8):
    """

    Args:
        observations (array): (num_modalities, time, num_obs)
        locations_matrix (array): (num_modalities, 2)
    """

    (
        agents,
        RG,
        LG,
    ) = (
        [],
        [],
        [],
    )
    observations = [observations]
    for n in range(max_levels):
        G = spm_space(locations_matrix)
        if n == 0 and num_controls > 0:
            # prepend action one_hots to first group
            G = [g + num_controls for g in G]
            G[0] = jnp.concatenate((jnp.arange(num_controls), G[0]))

        T = spm_time(observations[n].shape[1], dt)

        A = [None] * observations[n].shape[0]
        B = []
        A_dependencies = [None] * observations[n].shape[0]
        for g in range(len(G)):
            a, b = spm_structure_fast(observations[n][G[g]], dt)

            for m_relative, m_idx in enumerate(G[g]):
                A[m_idx] = a[m_relative]
                A_dependencies[m_idx] = [g]

            B += b

        RG.append(G)
        LG.append(locations_matrix)

        pdp = Agent(A=A, B=B, A_dependencies=A_dependencies, apply_batch=True, onehot_obs=True)
        agents.append(pdp)

        # observation dim size for next level
        ndim = max(max(pdp.num_states), max(pdp.num_controls))
        # we have to gather initial state and path as observations for the next level
        observations.append(jnp.zeros((len(G) * 2, len(T), ndim)))

        # Solve at the next timescale
        for t in range(len(T)):
            sub_horizon = T[t]
            for j in range(len(sub_horizon)):

                current_obs = observations[n].shape[0] * [None]
                for g in range(len(G)):
                    for m_g in range(len(G[g])):
                        # get a new observation from the n-th hierarchical level, the G[g] different modality indices for this group, the j-th timestep within `sub_horizon`.
                        # the [None,...] is to add a trivial batch dimension
                        obs_n_g_t = observations[n][G[g][m_g], sub_horizon[j], :][None, None, ...]
                        current_obs[G[g][m_g]] = jnp.copy(obs_n_g_t)

                # if we're beyond the first timestep, append observations_list to a growing list of historical observations
                if j > 0:
                    current_obs = jtu.tree_map(
                        lambda prev_o, new_o: jnp.concatenate([prev_o, new_o], 1), previous_obs, current_obs
                    )

                if j == 0:
                    qs = pdp.infer_states(current_obs, past_actions=None, empirical_prior=pdp.D, qs_hist=None)
                else:
                    qs = pdp.infer_states(current_obs, past_actions=None, empirical_prior=empirical_prior, qs_hist=qs)

                # fix to only one path for now?
                # how do we get the actual path? infer? plan?
                action = jnp.zeros(len(G), dtype=int)[None, ...]
                empirical_prior, qs_hist = pdp.infer_empirical_prior(action, qs)

                previous_obs = jtu.tree_map(
                    lambda x: jnp.copy(x), current_obs
                )  # set the current observation (and history) equal to the previous set of observations

                # print(f'Shape of state posterior over factor 0 at time {sub_horizon[j]}: {qs[0].shape}')
                print(f"Maximum probability state about factor 0 at time {sub_horizon[j]}: {jnp.argmax(qs[0][0,-1,:])}")

            ### How to deal with the 'even-odd' states and paths storage to follow:
            # Equalize their lengths by padding with zeros. For example, if hidden states are 16-dimensional and paths are 2 dimensional, then you just
            # make paths have 16 dims as well (pad with zeros). Then you'll still have (num_modalities, timesteps, 16) for O[n]

            # create function that marginalizes out the q(s_{t}) and q(s_{t+1}) from p(s_{t+1} | s_t, u_t) to get q(u_t)
            action_marginal_fn = lambda b, qs: factor_dot(b, qs, keep_dims=(2,))

            for g in range(len(G)):
                # infer q(paths) using the consecutive timesteps of qs_hist[g] (namely qs_hist[g][0, 0, :] and qs_hist[g][0, 1, :])
                q_u = vmap(action_marginal_fn)(pdp.B[g], [qs_hist[g][:, 0, :], qs_hist[g][:, 1, :]])

                # initial state (even indices)
                observations[n + 1] = observations[n + 1].at[2 * g, t, : qs[g].shape[-1]].set(qs_hist[g][0, 0, :])
                # path (odd indices)
                # observations[n + 1] = observations[n + 1].at[2 * g + 1, t, action[g]].set(1.0)
                observations[n + 1] = observations[n + 1].at[2 * g + 1, t, : q_u.shape[-1]].set(q_u[0])

        # coarse grain locations
        coarse_locations = []
        for g in range(len(G)):
            # append twice (for initial state and path)
            coarse_locations.append(jnp.mean(locations_matrix[G[g]], axis=0))
            coarse_locations.append(jnp.mean(locations_matrix[G[g]], axis=0))
        locations_matrix = jnp.stack(coarse_locations)

        if len(G) == 1:
            break

    return agents, RG, LG


def pad_to_same_size(arrays: list):
    """
    Pad arrays to the same size along the last dimension
    """
    max_size = max([a.shape[-1] for a in arrays])
    padded = [jnp.pad(a, ((0, 0), (0, max_size - a.shape[-1]))) for a in arrays]
    return padded


def infer(agents, observations, priors=None):
    """
    Infer the top level state given the observations and priors.
    Some observations can be masked out with uniform vectors if not yet fully observed.
    When priors is None, we use the (uniform) priors in the agent's D tensors.

    Args:
        agents (list): list of n agents, n the number of levels in the hieararchy
        observations (array): (num_modalities, n**T, num_observation_bins): observations of the lowest level
        priors (list): list of n priors, n the number of levels in the hierarchy

    Returns:
        Inferred top level state
    """
    if priors is None:
        # TODO broadcasting priors based on the number of expected timesteps required for inferring 1 top level state
        # currently assuming T=2
        # priors = [
        #     jnp.broadcast_to(
        #         jnp.asarray(agents[n].D), (len(agents[n].D), 2 ** (abs(n - len(agents) + 1)), agents[n].D[0].shape[-1])
        #     )
        #     for n in range(len(agents))
        # ]

        # TODO not every agent has the same number of states, so we need to pad the priors to the same size
        priors = []
        for n in range(len(agents)):
            Ds = jnp.stack(pad_to_same_size(agents[n].D))
            priors.append(jnp.broadcast_to(Ds, (len(agents[n].D), 2 ** (abs(n - len(agents) + 1)), Ds.shape[-1])))

    for n in range(len(agents)):
        # infer states for each observation

        # convert observations array to a list or arrays (modalities), and make time the batch dimension to vmap over
        # TODO not considering actual batch dimension here
        o = [observations[i, :, :] for i in range(observations.shape[0])]

        priors_n = [priors[n][i, :, : agents[n].D[i].shape[-1]] for i in range(priors[n].shape[0])]

        # doesn't auto-broadcast to batch, call inference method vmapped ourselves
        # qs = agents[n].infer_states(o, past_actions=None, empirical_prior=priors, qs_hist=None)
        infer = partial(run_factorized_fpi, A_dependencies=agents[n].A_dependencies, num_iter=agents[n].num_iter)

        if priors_n[0].shape[0] != o[0].shape[0]:
            # longer timesequence of observations is given, need to broadcast priors
            k = o[0].shape[0] // priors_n[0].shape[0]
            priors_n = jtu.tree_map(
                lambda x: jnp.broadcast_to(x, (k, x.shape[0], x.shape[1])).reshape(o[0].shape[0], x.shape[1]), priors_n
            )

        qs = vmap(infer, in_axes=(None, 0, 0))(jtu.tree_map(lambda x: x[0], agents[n].A), o, priors_n)

        if n == len(agents) - 1:
            # reached the top level, no more paths to infer?, return this? (and only this?)
            return qs

        # now infer paths for each subsequence of T (= 2)
        q0 = jtu.tree_map(lambda x: x[::2, :], qs)
        q1 = jtu.tree_map(lambda x: x[1::2, :], qs)

        D = q0
        E = []
        # TODO make this a method instead of this loop?
        action_marginal_fn = lambda b, qs: factor_dot(b, qs, keep_dims=(2,))
        for g in range(len(agents[n].B)):
            E.append(vmap(action_marginal_fn, in_axes=(None, 0))(agents[n].B[g][0], [q0[g], q1[g]]))

        ndim = max([d.shape[-1] for d in D] + [e.shape[-1] for e in E])

        # pad D and E to be same trailing dim
        D = jtu.tree_map(lambda x: jnp.pad(x, ((0, 0), (0, ndim - x.shape[-1]))), D)
        E = jtu.tree_map(lambda x: jnp.pad(x, ((0, 0), (0, ndim - x.shape[-1]))), E)

        # now interleave init state 0, path 0, start state 1, path 1, etc.
        interleaved = E + D
        interleaved[::2] = D
        interleaved[1::2] = E
        observations = jnp.asarray(interleaved)


def predict(agents, D=None, E=None, num_steps=1):
    """
    Infer the top level state given the observations and priors.
    Some observations can be masked out with uniform vectors if not yet fully observed.
    When priors is None, we use the (uniform) priors in the agent's D tensors.

    Args:
        agents (list): list of n agents, n the number of levels in the hieararchy
        D (list): list of initial state factors at the top level
        E (list): list of path at the top level

    Returns:
        Predicted states and observations for all levels in the hierarchy
    """

    n = len(agents) - 1

    beliefs = [
        None,
    ] * (len(agents))
    observations = [
        None,
    ] * (len(agents))

    # add time dimension, so qs[f] has shape (batch_dim, 1, num_states)
    if D is None:
        D = agents[n].D
    qs = jtu.tree_map(lambda x: jnp.expand_dims(x, 1), D)

    # unroll highest level
    expected_state = partial(compute_expected_state, B_dependencies=agents[n].B_dependencies)

    for _ in range(num_steps):
        # extract the last timestep, such tthat qs_last[f] has shape (batch_dim, num_states)
        qs_last = jtu.tree_map(lambda x: x[:, -1, ...], qs)
        # this computation of the predictive prior is correct only for fully factorised Bs.
        pred = vmap(expected_state)(qs_last, agents[n].B, E)
        # pred, qs  = agents[n].infer_empirical_prior(E, qs)
        # stack in time dimension
        qs = jtu.tree_map(
            lambda x, y: jnp.concatenate([x, jnp.expand_dims(y, 1)], 1),
            qs,
            pred,
        )
    # qs[f] will have shape (batch_dim, num_steps+1, num_states)

    qs_stacked = jtu.tree_map(lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], x.shape[2])), qs)
    beliefs[n] = qs_stacked

    # generate outcomes of highest level, vmap over time
    expected_obs = partial(compute_expected_obs, A_dependencies=agents[n].A_dependencies)
    A_stacked = jtu.tree_map(
        lambda x: jnp.broadcast_to(x, (qs_stacked[0].shape[0], x.shape[1], x.shape[2])), agents[n].A
    )
    qo = vmap(expected_obs)(qs_stacked, A_stacked)

    observations[n] = qo

    while n > 0:
        qo = observations[n]

        n -= 1
        agent = agents[n]

        # split this in initial state "D" and path "E"
        DD = qo[::2]
        for i in range(len(agent.B)):
            DD[i] = DD[i][:, : agent.B[i].shape[1]]
        EE = jtu.tree_map(lambda x: jnp.argmax(x, axis=1), qo[1::2])

        # unroll path and get beliefs qs at level n
        # TODO repeat if dt > 2
        expected_state = partial(compute_expected_state, B_dependencies=agents[n].B_dependencies)
        B_stacked = jtu.tree_map(
            lambda x: jnp.broadcast_to(x, (DD[0].shape[0], x.shape[1], x.shape[2], x.shape[3])), agents[n].B
        )
        pred = vmap(expected_state)(DD, B_stacked, EE)

        # stack in time dimension
        qs = jtu.tree_map(
            lambda x, y: jnp.concatenate([jnp.expand_dims(x, 1), jnp.expand_dims(y, 1)], 1),
            DD,
            pred,
        )

        qs_stacked = jtu.tree_map(lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], x.shape[2])), qs)
        beliefs[n] = qs_stacked

        # now generate outcomes of level n, vmap over time
        expected_obs = partial(compute_expected_obs, A_dependencies=agents[n].A_dependencies)
        A_stacked = jtu.tree_map(
            lambda x: jnp.broadcast_to(x, (qs_stacked[0].shape[0], x.shape[1], x.shape[2])), agents[n].A
        )
        qo = vmap(expected_obs)(qs_stacked, A_stacked)

        observations[n] = qo

    observations = [jnp.asarray(o) for o in observations]
    beliefs = [jnp.stack(pad_to_same_size(b)) for b in beliefs]
    return observations, beliefs


def learn_transitions(qs, actions=None, B_dependencies=None, pB=None):
    """
    qs: a list of jax.numpy arrays of shape [(n_time, n_states) for f in factors]
    """

    n_states = qs[0].shape[-1]
    n_time = qs[0].shape[0] - 1
    print(f"Fitting for {n_states} states and {n_time} timesteps.")

    if pB is None:
        pB = [1e-3 * jnp.ones((n_states, n_states, 1)) for _ in range(len(qs))]

    if actions is None:
        # No actions = the zero action at each timestep
        actions = jnp.zeros((n_time, len(qs)))

    if B_dependencies is None:
        B_dependencies = [[i] for i in range(len(qs))]

    # Map the qs's to something that can be used for learning.
    beliefs = []

    # take all the states but the last one
    qs_f_prev = jtu.tree_map(lambda x: x[:-1], qs)
    # take all the states but the first one
    qs_f = jtu.tree_map(lambda x: x[1:], qs)

    for f in range(len(pB)):

        # Extract factor
        q_f = jnp.array(qs_f[f].tolist())
        q_prev_f = [jnp.array(qs_f_prev[fi].tolist()) for fi in B_dependencies[f]]
        beliefs.append([q_f, *q_prev_f])

    qB, E_qB = update_state_transition_dirichlet(pB, beliefs, actions, num_controls=[b.shape[-1] for b in pB], lr=1)

    norm = lambda x: jnp.divide(
        jnp.clip(x, a_min=1e-8),
        jnp.clip(x, a_min=1e-8).sum(axis=1, keepdims=True),
    )

    E_qB = jtu.tree_map(norm, qB)

    return qB, E_qB


if __name__ == "__main__":

    path_to_file = "examples/structure_learning/dove.mp4"

    # Read in the video file as tensor (num_frames, width, height, channels)
    frames = read_frames_from_mp4(path_to_file)

    # Map the RGB image to discrete outcomes
    # Observations are list[list[array]] -> num modalities, time-steps, num_discrete_bins
    # Location matrix is num_modalities x 2 (width, height)
    # Group indices is num_modalities
    # sv_discrete_axis num_modalities x num_discrete_bins
    # V_per_patch num_patches, num_pixels_per_patch x 11?
    (observations, locations_matrix, group_indices, sv_discrete_axis, V_per_patch), patch_indices = map_rgb_2_discrete(
        frames, tile_diameter=32, n_bins=9, sv_thr=(1.0 / 5.0)
    )

    # convert list of list of observation one-hots into an array of size (num_modalities, timesteps, num_obs)
    observations = jnp.asarray(observations)

    # Run structure learning on the observations
    agents, RG, LB = spm_mb_structure_learning(observations, locations_matrix, max_levels=8)

    plt.imshow(frames[0])
    for locations_matrix in LB:
        plt.scatter(locations_matrix[:, 0], locations_matrix[:, 1], c="r")
    plt.show()

    for A_m in A[0]:
        print(A_m[0].shape)
        print(A_m)
    for B_m in B[0]:
        print(B_m[0].shape)
        print(B_m)

    G = spm_space(locations_matrix)
    print(G)

    ims = []

    # Map the discrete outcomes back to RGB
    observations = jnp.array(observations)
    video = jnp.zeros(frames.shape)
    for t in range(observations.shape[1]):
        img = map_discrete_2_rgb(
            observations[:, t, :],
            locations_matrix,
            group_indices,
            sv_discrete_axis,
            V_per_patch,
            patch_indices,
            frames.shape[-3:],
        )

        # this reconstructs 2 frames
        for i in range(2):
            im = img[i, ...]
            # transform back to RGB
            im = jnp.transpose(im, (1, 2, 0))
            im /= 255
            im = jnp.clip(im, 0, 1)
            im = (255 * im).astype(onp.uint8)

            gt = frames[t * 2 + i]
            ims.append(onp.hstack([im, gt]))

    imageio.mimsave("reconstruction.gif", ims)
