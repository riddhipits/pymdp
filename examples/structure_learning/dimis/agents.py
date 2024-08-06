import equinox as eqx
import jax.numpy as jnp
import numpy as np
import jax.random as jr

from typing import Dict, List, Optional
from jax import jit, vmap, nn, tree_util as jtu
from jaxtyping import Array
from utils import Patch

from structure_learning import fast_structure_learning, forward_backward_algo, batch_mult
from pymdp.jax.agent import Agent as AIFAgent
from functools import partial

class SLAgent(eqx.Module):
    As: Dict[int, Array]
    Bs: Dict[int, Array]
    # pAs: Dict[int, List[Array]]
    # pBs: Dict[int, List[Array]]
    H: Optional[Array]
    I: Optional[List[Array]]
    patchers: Dict[int, Patch]
    number_of_actions: int = eqx.field(static=True)
    likelihood_noise: bool = eqx.field(static=True)
    num_states: Dict[int, List[int]]
    num_paths: Dict[int, List[int]]
    max_state_paths: Dict[int, List[int]] = eqx.field(static=True)
    
    def __init__(self, image_size, patch_size, number_of_actions, params=None, likelihood_noise=False):
        self.number_of_actions = number_of_actions
        self.likelihood_noise = likelihood_noise

        if params is None:
            self.initialize_states_and_likelihoods(image_size, patch_size)
        else:
            (self.As, self.Bs, self.H, self.patchers, self.num_states, self.num_paths, self.max_state_paths) = params

        level = len(self.As) - 1
        loc_agent = self.make_agents_for_level(level=level, policy_len=1)
        self.I = loc_agent.I

    def initialize_states_and_likelihoods(self, image_size, patch_size):
        image = jnp.zeros((1,) + image_size)
        num_pixels = np.prod(image_size)
        actions = nn.one_hot(jnp.arange(self.number_of_actions), num_pixels).reshape((-1, 1,) + image_size)

        joint = vmap(lambda x, y: jnp.concatenate([x, y], -3), in_axes=(None, 0))(image, actions)
        (self.As, self.Bs, self.H, self.patchers, self.num_states, self.num_paths, self.max_state_paths) = fast_structure_learning(
            jnp.expand_dims(joint, 1), 
            image_size=image_size, 
            patch_size=patch_size
        )

    def make_agents_for_level(self, *, level, policy_len=1, D=None, E=None):

        # max values for outputs
        max_ns, max_np = self.max_state_paths[level]

        # max values for inputs
        # _max_ns, _ = self.max_state_paths[level + 1]

        num_outcomes = max(max_ns, max_np)

        As = self.As[level]
        Bs = self.Bs[level]
        num_states = jnp.array(self.num_states[level])
        num_paths = jnp.array(self.num_paths[level])
        A = As.reshape(As.shape[:2] + (-1,))
        num_mods = A.shape[-1] # number of modalities
        # pAs = jnp.stack(
        #         jtu.tree_map(
        #             lambda x: jnp.pad(x, (0, _max_ns - len(x))), self.pAs[level]
        #         )
        #     )

        # if self.likelihood_noise:
        #     z = pAs[:, None]

        #     alpha0_1 = (jnp.arange(num_outcomes) < max_ns) / max_ns
        #     alpha0_2 = (jnp.arange(num_outcomes) < max_np) / max_np
        #     patch_size = self.patchers[level].patch_size
        #     alpha0 = jnp.ones(patch_size + (1,)) * jnp.stack([alpha0_1, alpha0_2], -1)[:, None, None]
        #     alpha0 = alpha0.reshape(num_outcomes, -1, 1)
        # else:
        #     z = pAs[:, None] > 0.
        #     alpha0 = jnp.expand_dims((1 - z), 1)

        # _pAs = [ z * nn.one_hot(A[..., i], num_outcomes, axis=1) + alpha0[:, i] for i in range(num_mods) ]

        # A = jtu.tree_map(
        #     lambda x: x / x.sum(1, keepdims=True), _pAs
        # )

        A = [nn.one_hot(A[..., i], num_outcomes, axis=1) for i in range(num_mods) ]

        B = [Bs]
        C = [jnp.zeros(a.shape[:2]) for a in A]

        if D is None:
            d = jnp.zeros_like(num_states)
            D = jnp.zeros(Bs.shape[:2])
            ns = D.shape[-1]
            for _ in range(ns):
                D += nn.one_hot(d, ns)
                d += 1
                d = jnp.clip(d, a_max=num_states - 1)
            D = [D / D.sum(-1, keepdims=True)]
        else:
            D = [D]

        if E is None:
            e = jnp.zeros_like(num_paths)
            np = Bs.shape[-1]
            max_policies = np ** policy_len
            num_policies = num_paths ** policy_len
            E = jnp.zeros((len(e), max_policies))
            for _ in range(max_policies):
                E += nn.one_hot(e, max_policies)
                e += 1
                e = jnp.clip(e, a_max=num_policies - 1)

        pA = None
        pB = None
        if level == len(self.As) - 1:
            #TODO: instead of H pass I which is constant
            try:
                I = self.I
                H = None
            except:
                I = None
                H = [self.H] if self.H is not None else None
            
            use_inductive = True
        else:
            H = None
            I = None
            use_inductive = False

        agents = AIFAgent(
                    A,
                    B,
                    C,
                    D,
                    E,
                    pA,
                    pB,
                    H=H,
                    I=I,
                    gamma=8.0,
                    inductive_depth=256,
                    inductive_threshold=0.01,
                    inductive_epsilon=1e-5,
                    policy_len=policy_len,
                    use_utility=False,
                    use_inductive=use_inductive,
                    use_states_info_gain=False,
                )
        
        return agents

    @jit
    def top_down_prediction(self, D=None, E=None, T=1):
        L = len(self.As)
        priors = {L: {'D': D[None], 'E': E[None]}}
        for level in range(L - 1, -1, -1):
            make_patches = self.patchers[level]
            max_ns, max_np = self.max_state_paths[level]
            if level < L - 1:
                agent = self.make_agents_for_level(
                    level=level, policy_len=1
                )
                beliefs = []
                for d, e in zip(D, E):
                    agent = eqx.tree_at(lambda x: (x.E, x.D), agent, (e, d))
                    beliefs.append(d)
                    q_pi, _ = agent.infer_policies([jnp.expand_dims(beliefs[-1], 1)])
                    beliefs.append(batch_mult(batch_mult(agent.B[0], q_pi), beliefs[-1]))
            else:
                agent = self.make_agents_for_level(level=level, policy_len=1, D=D, E=E)
                beliefs = agent.D
                e = agent.E
                q_pi, _ = agent.infer_policies([jnp.expand_dims(beliefs[-1], 1)])

            qs = jnp.stack(beliefs, -2)
            A = jnp.stack(agent.A, -1)
            pred_outcomes = vmap(lambda x, y: y @ x)(A, qs)
            pred_outcomes = pred_outcomes.reshape(pred_outcomes.shape[:3] + make_patches.patch_size + (make_patches.in_chans,))
            pred_outcomes = jnp.moveaxis(pred_outcomes, 0, 2)
            batch_shape = pred_outcomes.shape[:2]
            img_shape = pred_outcomes.shape[2:]
            pred_outcomes = vmap(make_patches.inverse)(pred_outcomes.reshape((-1,) + img_shape))
            D = jnp.moveaxis(pred_outcomes[:, 0].reshape(batch_shape + (-1,)), 0, -1)[..., :max_ns]
            E = jnp.moveaxis(pred_outcomes[:, 1].reshape(batch_shape + (-1,)), 0, -1)[..., :max_np]
            priors[level] = {'D': D, 'E': E}

        return priors

    def bottom_up_inference(self, obs, masks, priors):
        L = len(priors) - 1
        for level in range(L):
            K = len(obs)
            agent = self.make_agents_for_level(
                    level=level, policy_len=1
                )
            D, E = priors[level + 1]['D'], priors[level + 1]['E']

            if masks is None:
                masks = jnp.ones(obs.shape[:-1], dtype=bool)

            obs = jnp.pad(obs, ((0, K % 2), (0, 0), (0, 0), (0, 0)), constant_values=1.)
            obs = obs.reshape((-1, 2,) + obs.shape[1:])
            
            masks = jnp.pad(masks, ((0, K % 2), (0, 0), (0, 0)))
            masks = masks.reshape((-1, 2,) + masks.shape[1:])

            fba = vmap(partial(forward_backward_algo, agent))
            k = obs.shape[0]
            beliefs = fba(obs, masks, D[:k, None], E[:k, None])

            if level < L - 1:
                max_ns, max_np = self.max_state_paths[level+1]
                no = max(max_ns, max_np)
                obs = [
                    jnp.pad(beliefs[0][:, 0], ((0, 0), (0, 0), (0, no - max_ns))),
                    jnp.pad(beliefs[1], ((0, 0), (0, 0), (0, no - max_np)))
                ]
                obs = jnp.moveaxis(jnp.stack(obs, -3), -1, 1)
                make_patches = self.patchers[level+1]
                obs = obs.reshape(obs.shape[:-1] + make_patches.img_size)
                batch_shape = obs.shape[:2]
                obs = vmap(make_patches)(obs.reshape((-1,) + obs.shape[-3:]))
                num_patches = obs.shape[1]
                obs = jnp.moveaxis(obs.reshape(batch_shape + (num_patches, -1)), 1, -1)
                masks = None
            
        # return top level Ds, and E
        top_D_current =  beliefs[0][0, 0]
        top_D_next = beliefs[0][0, 1]
        top_E  = beliefs[1]
        return top_D_current, top_D_next,  top_E

    @jit
    def select_action(self, obs, priors, *, key, alpha=1.0, mask_all=False):
        make_patches = self.patchers[0]
        patched_obs = vmap(make_patches)(obs)
        
        mask = jnp.ones_like(patched_obs[:-1], dtype=bool)
        mask_last = jnp.concatenate(
            [
                jnp.ones_like(patched_obs[0, ..., :-1], dtype=bool),
                jnp.zeros_like(patched_obs[0, ..., -1:], dtype=bool)
            ],
            -1
        )
        
        mask = jnp.concatenate([mask, mask_last[None]], 0)
        mask = mask * (1 - mask_all) + mask_all * jnp.zeros_like(mask, dtype=bool)
        no = max(self.max_state_paths[0])
        batch_shape = patched_obs.shape[:2]
        O = nn.one_hot(patched_obs.reshape(batch_shape + (-1,)), no)
        mask = mask.reshape(batch_shape + (-1,))

        # perform inference with masked observations (action is not observed)
        # and return new beliefs about states at the top level, for the current 
        # and the next step. Use current D, and E  to generate top down predictions
        # and readout actions
        D_current, D_next, E = self.bottom_up_inference(O, mask, priors)

        #TODO: return priors also for the next time step?
        new_priors = self.top_down_prediction(D=D_current, E=E)
        k = len(obs) - 1
        control = new_priors[0]['E'][k, :self.number_of_actions, -1]
        return jr.categorical(key, jnp.log(control)), D_next