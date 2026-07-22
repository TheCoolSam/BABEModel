"""
model.py - SocialNetworkModel for the Dynamic-BABE ABM

Implementations of the full model:
  * Barabasi-Albert network generation
  * Per-tick agent interaction loop
  * NOVEL: Dyadic Trust co-evolution (symmetric)
  * Bridge Algorithm interception (toggle)
  * Revenue step-function computation
  * Echo Chamber detection metrics
  * Mesa DataCollector for all KPIs

Deterministic: accepts a seed that flows into numpy, networkx,
and the Mesa scheduler so that every run is exactly reproducible.
"""

from typing import Optional, List, Tuple

import networkx as nx
import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation

import config as cfg
from agent import SocialAgent


# ====================================================================
# Helper functions used by the DataCollector
# ====================================================================

def _active_agents(model):
    """Return only agents that have not churned."""
    return [a for a in model.schedule.agents if a.active]


def compute_polarization(model):
    """
    Global polarization sigma - root-mean-square of per-dimension
    standard deviations across all active opinion vectors.

    This avoids the "Data Flattening" trap: np.mean(opinion)
    would cancel out multidimensional extremism (e.g. [+1,-1]
    looks like 0).  Instead we measure spread in EACH dimension
    and combine via RMS (Root Mean Square).
    """
    active = _active_agents(model)
    if len(active) < 2:
        return 0.0
    # Stack into (n_agents, n_issues) matrix
    opinions = np.array([a.opinion for a in active])
    per_dim_std = np.std(opinions, axis=0)          # std per issue dimension
    return float(np.sqrt(np.mean(per_dim_std ** 2))) # RMS of per-dim stds


def compute_revenue(model):
    """
    Revenue step-function:
      Base = Sum(Active Users) * Base Ad Rate
      If sigma >= cliff -> apply unsafe ad multiplier (brand-safety penalty).
    """
    active_count = len(_active_agents(model))
    sigma = compute_polarization(model)
    unsafe = getattr(model, "unsafe_ad_multiplier", cfg.UNSAFE_AD_MULTIPLIER)
    multiplier = (
        cfg.SAFE_AD_MULTIPLIER
        if sigma < cfg.POLARIZATION_CLIFF
        else unsafe
    )
    return active_count * cfg.BASE_AD_RATE * multiplier


def compute_churn_rate(model):
    """Fraction of the original population that has churned."""
    total = len(model.schedule.agents)
    churned = sum(1 for a in model.schedule.agents if not a.active)
    return churned / total if total else 0.0


def compute_avg_frustration(model):
    """Mean frustration across *active* agents only."""
    active = _active_agents(model)
    if not active:
        return 0.0
    return float(np.mean([a.frustration for a in active]))


def compute_active_users(model):
    return len(_active_agents(model))


# ====================================================================
# NOVEL: Trust metrics
# ====================================================================

def compute_mean_trust(model):
    """Mean trust across all edges in the network."""
    if not model.enable_trust:
        return cfg.TRUST_INITIAL
    trusts = [
        d.get('trust', cfg.TRUST_INITIAL)
        for _, _, d in model.G.edges(data=True)
    ]
    return float(np.mean(trusts)) if trusts else cfg.TRUST_INITIAL


def compute_ingroup_trust(model):
    """
    Mean trust on edges where both endpoints share the same opinion pole.

    'Same pole' = sign of mean opinion is the same. This captures the
    intuitive notion of 'ingroup' in a bipolar opinion landscape.
    """
    if not model.enable_trust:
        return cfg.TRUST_INITIAL
    trusts = []
    for u, v, d in model.G.edges(data=True):
        a_u = model.G.nodes[u].get("agent")
        a_v = model.G.nodes[v].get("agent")
        if a_u is None or a_v is None:
            continue
        if not a_u.active or not a_v.active:
            continue
        # Same pole = same sign of mean opinion.
        # Note: sign(0)=0 is a distinct zero pole (rare under bipolar init).
        if np.sign(a_u.opinion.mean()) == np.sign(a_v.opinion.mean()):
            trusts.append(d.get('trust', cfg.TRUST_INITIAL))
    return float(np.mean(trusts)) if trusts else 0.0


def compute_outgroup_trust(model):
    """
    Mean trust on edges where endpoints are on OPPOSITE opinion poles.

    The ratio ingroup/outgroup trust quantifies 'trust segregation' —
    a novel KPI bridging network science and social psychology.
    """
    if not model.enable_trust:
        return cfg.TRUST_INITIAL
    trusts = []
    for u, v, d in model.G.edges(data=True):
        a_u = model.G.nodes[u].get("agent")
        a_v = model.G.nodes[v].get("agent")
        if a_u is None or a_v is None:
            continue
        if not a_u.active or not a_v.active:
            continue
        # Opposite poles
        if np.sign(a_u.opinion.mean()) != np.sign(a_v.opinion.mean()):
            trusts.append(d.get('trust', cfg.TRUST_INITIAL))
    return float(np.mean(trusts)) if trusts else 0.0


# ====================================================================
# NOVEL: Echo Chamber detection
# ====================================================================

def compute_opinion_clustering(model):
    """
    Opinion-Weighted Clustering Coefficient.

    Measures the fraction of each node's triangles where all three
    vertices share the same opinion pole.  High values indicate
    structurally-clustered echo chambers.

    Returns global average across active agents.
    """
    if not cfg.ENABLE_ECHO_CHAMBER_METRICS:
        return 0.0

    active = _active_agents(model)
    if len(active) < 3:
        return 0.0

    G = model.G
    scores = []

    for agent in active:
        nid = agent.unique_id
        neighbours = list(G.neighbors(nid))
        if len(neighbours) < 2:
            scores.append(0.0)
            continue

        # This agent's pole
        pole_i = np.sign(agent.opinion.mean())

        total_triangles = 0
        same_pole_triangles = 0

        for idx_a in range(len(neighbours)):
            for idx_b in range(idx_a + 1, len(neighbours)):
                na, nb = neighbours[idx_a], neighbours[idx_b]
                if not G.has_edge(na, nb):
                    continue
                # Triangle found: (nid, na, nb)
                total_triangles += 1

                a_na = G.nodes[na].get("agent")
                a_nb = G.nodes[nb].get("agent")
                if a_na is None or a_nb is None:
                    continue
                if not a_na.active or not a_nb.active:
                    continue

                pole_a = np.sign(a_na.opinion.mean())
                pole_b = np.sign(a_nb.opinion.mean())

                if pole_i == pole_a == pole_b:
                    same_pole_triangles += 1

        if total_triangles > 0:
            scores.append(same_pole_triangles / total_triangles)
        else:
            scores.append(0.0)

    return float(np.mean(scores))


def compute_trust_segregation(model):
    """
    Trust Segregation Ratio = ingroup_trust / outgroup_trust.

    Values > 1 indicate trust is concentrated within opinion clusters
    (echo chamber behaviour). Values ≈ 1 indicate uniform trust.
    Returns 0.0 if outgroup trust is zero (complete segregation).
    """
    if not model.enable_trust:
        return 1.0  # No trust system → no segregation
    ingroup = compute_ingroup_trust(model)
    outgroup = compute_outgroup_trust(model)
    if outgroup < 1e-9:
        return float('inf') if ingroup > 1e-9 else 1.0
    return ingroup / outgroup


# ====================================================================
# Model
# ====================================================================

class SocialNetworkModel(Model):
    """
    Dynamic-BABE (Biased Assimilation & Behavioral Entrenchment) Model.

    Parameters
    ----------
    num_agents      : population size  (default from config)
    ba_m            : Barabasi-Albert edge parameter
    enable_bridge   : activate the Bridge Algorithm dampener
    seed            : master RNG seed for full reproducibility
    """

    def __init__(
        self,
        num_agents=cfg.NUM_AGENTS,
        ba_m=cfg.BA_EDGE_PARAM,
        enable_bridge=cfg.ENABLE_BRIDGE,
        enable_trust=cfg.ENABLE_TRUST,
        seed=cfg.RANDOM_SEED,
        bridge_efficacy=None,
        bridge_healing_bonus=None,
        trust_decay=None,
        unsafe_ad_multiplier=None,
        topology=None,
        ws_k=None,
        ws_p=None,
        bridge_mode=None,
        disable_trust_decay=None,
        opinion_model=None,
        hk_epsilon=None,
    ):
        super().__init__()

        # Reproducibility
        # Partner choice / bridge draws use numpy Generator(seed).
        # Mesa RandomActivation shuffle uses model.random, which in Mesa 2.x
        # is keyed off self._seed — set _seed BEFORE creating the scheduler
        # so activation order is deterministic for this seed. Do not reorder
        # these assignments without re-validating batch reproducibility.
        self.rng = np.random.default_rng(seed)
        self._seed = seed

        # Parameters (instance overrides keep multiprocessing workers isolated)
        self.num_agents = num_agents
        self.enable_bridge = enable_bridge
        self.enable_trust = enable_trust
        self.bridge_efficacy = (
            cfg.BRIDGE_EFFICACY if bridge_efficacy is None else bridge_efficacy
        )
        self.bridge_healing_bonus = (
            cfg.BRIDGE_HEALING_BONUS
            if bridge_healing_bonus is None
            else bridge_healing_bonus
        )
        self.trust_decay = cfg.TRUST_DECAY if trust_decay is None else trust_decay
        self.unsafe_ad_multiplier = (
            cfg.UNSAFE_AD_MULTIPLIER
            if unsafe_ad_multiplier is None
            else unsafe_ad_multiplier
        )
        self.topology = (topology or getattr(cfg, "TOPOLOGY", "ba")).lower()
        self.ws_k = cfg.WS_K if ws_k is None else ws_k
        self.ws_p = cfg.WS_P if ws_p is None else ws_p
        self.bridge_mode = (
            bridge_mode or getattr(cfg, "BRIDGE_MODE", "intercept")
        ).lower()
        self.disable_trust_decay = (
            getattr(cfg, "DISABLE_TRUST_DECAY", False)
            if disable_trust_decay is None
            else bool(disable_trust_decay)
        )
        self.opinion_model = (
            opinion_model or getattr(cfg, "OPINION_MODEL", "dynamic_babe")
        ).lower()
        self.hk_epsilon = (
            cfg.HK_EPSILON if hk_epsilon is None else hk_epsilon
        )

        # Network topology
        g_seed = int(self.rng.integers(0, 2**31))
        if self.topology in ("watts_strogatz", "ws"):
            k = int(self.ws_k)
            if k % 2 == 1:
                k += 1
            k = max(2, min(k, num_agents - 1 if num_agents > 1 else 2))
            self.G = nx.watts_strogatz_graph(
                n=num_agents, k=k, p=float(self.ws_p), seed=g_seed
            )
        else:
            self.G = nx.barabasi_albert_graph(
                n=num_agents, m=ba_m, seed=g_seed
            )

        # ------------------------------------------------------------ #
        # NOVEL: Initialise symmetric dyadic trust on every edge
        # NOTE: Currently symmetric (T_ij = T_ji). Future work should
        #       explore asymmetric trust where each direction stores
        #       its own weight.
        # ------------------------------------------------------------ #
        if self.enable_trust:
            for u, v in self.G.edges():
                self.G[u][v]['trust'] = cfg.TRUST_INITIAL

        # Scheduler (after _seed so shuffle stream matches seed)
        self.schedule = RandomActivation(self)

        # Populate
        for node_id in self.G.nodes():
            agent = SocialAgent(node_id, self, self.rng)
            self.schedule.add(agent)
            self.G.nodes[node_id]["agent"] = agent

        # Data collection — build reporters dict dynamically
        model_reporters = {
            "Polarization": compute_polarization,
            "Revenue": compute_revenue,
            "Churn_Rate": compute_churn_rate,
            "Avg_Frustration": compute_avg_frustration,
            "Active_Users": compute_active_users,
            # NOVEL: trust reporters (return defaults when trust disabled)
            "Mean_Trust": compute_mean_trust,
            "Ingroup_Trust": compute_ingroup_trust,
            "Outgroup_Trust": compute_outgroup_trust,
            "Trust_Segregation": compute_trust_segregation,
            # NOVEL: echo chamber reporter
            "Opinion_Clustering": compute_opinion_clustering,
        }

        self.datacollector = DataCollector(
            model_reporters=model_reporters,
            agent_reporters={
                # Extremity captures multidimensional
                # radicalism without the flattening trap of np.mean().
                # A "Chaos Agent" at [+1, -1] scores 1.0 (radical),
                # NOT 0.0 (false moderate).
                "Opinion_Extremity": lambda a: float(
                    np.linalg.norm(a.opinion) / np.sqrt(len(a.opinion))
                ),
                "Frustration": lambda a: a.frustration,
                "Active": lambda a: a.active,
                "Beta": lambda a: a.beta,
            },
        )

        # Bookkeeping
        self.running = True
        self.datacollector.collect(self)  # Collect initial state (t = 0)

    def interact(self, agent):
        """
        Select a random active neighbour and run the configured
        opinion-update pipeline (Dynamic-BABE, HK, or degroot_bias).
        """
        neighbours = list(self.G.neighbors(agent.unique_id))
        if not neighbours:
            return

        active_neighbours = [
            self.G.nodes[nid]["agent"]
            for nid in neighbours
            if self.G.nodes[nid]["agent"].active
        ]
        if not active_neighbours:
            return

        k = min(cfg.INTERACTIONS_PER_STEP, len(active_neighbours))
        partners = self.rng.choice(active_neighbours, size=k, replace=False)

        for partner in partners:
            if self.opinion_model == "hk":
                self._process_hk(agent, partner)
            elif self.opinion_model == "degroot_bias":
                self._process_degroot_bias(agent, partner)
            else:
                self._process_interaction(agent, partner)

    def _process_hk(self, agent_i, agent_j):
        """Hegselmann-Krause-style bounded confidence (no SJT/bridge/trust/churn)."""
        dist = float(np.linalg.norm(agent_i.opinion - agent_j.opinion)) / np.sqrt(
            len(agent_i.opinion)
        )
        if dist < self.hk_epsilon:
            agent_i.opinion = np.clip(
                0.5 * (agent_i.opinion + agent_j.opinion), -1.0, 1.0
            )

    def _process_degroot_bias(self, agent_i, agent_j):
        """Lightweight biased assimilation without SJT zones, bridge, or trust."""
        alignment = agent_i.compute_alignment(agent_j)
        w_ij = agent_i.compute_influence_weight(alignment)
        if w_ij > 0:
            mu = w_ij / (1.0 + agent_i.beta)
            agent_i.opinion = np.clip(
                agent_i.opinion + mu * (agent_j.opinion - agent_i.opinion),
                -1.0,
                1.0,
            )
        else:
            repulsion = abs(w_ij) / (1.0 + agent_i.beta)
            away = agent_i.opinion - agent_j.opinion
            agent_i.opinion = np.clip(
                agent_i.opinion + repulsion * away, -1.0, 1.0
            )

    def _process_interaction(self, agent_i, agent_j):
        """Full Dynamic-BABE interaction pipeline (with bridge-mode ablations)."""
        alignment = agent_i.compute_alignment(agent_j)
        w_ij = agent_i.compute_influence_weight(alignment)

        if self.enable_trust:
            trust = agent_i.get_trust(agent_j)
            w_ij = w_ij + cfg.TRUST_INFLUENCE * trust

        zone = SocialAgent.classify_zone(w_ij)

        if self.enable_bridge and zone == 3:
            if self.rng.random() < self.bridge_efficacy:
                if self.bridge_mode == "heal_only":
                    agent_i.frustration = max(
                        0, agent_i.frustration - self.bridge_healing_bonus
                    )
                elif self.bridge_mode == "drop":
                    agent_i.frustration = max(
                        0, agent_i.frustration - self.bridge_healing_bonus
                    )
                    agent_i.check_churn()
                    return
                else:
                    w_ij = 0.0
                    zone = 2
                    agent_i.frustration = max(
                        0, agent_i.frustration - self.bridge_healing_bonus
                    )

        if zone == 1:
            agent_i.assimilate(agent_j, w_ij)
        elif zone == 2:
            pass
        elif zone == 3:
            agent_i.backfire(agent_j, w_ij)

        if self.enable_trust:
            edge = self.G[agent_i.unique_id][agent_j.unique_id]
            current_trust = edge.get("trust", cfg.TRUST_INITIAL)
            if zone == 1:
                new_trust = min(1.0, current_trust + cfg.TRUST_GAIN)
            elif zone == 3:
                new_trust = max(0.0, current_trust - cfg.TRUST_LOSS)
            else:
                new_trust = current_trust
            edge["trust"] = new_trust

        agent_i.check_churn()

    def _decay_trust(self):
        """Passive trust decay on all edges each step (unless disabled)."""
        if (
            not self.enable_trust
            or self.disable_trust_decay
            or self.trust_decay <= 0
        ):
            return
        decay_factor = 1.0 - self.trust_decay
        for u, v in self.G.edges():
            self.G[u][v]["trust"] = self.G[u][v].get(
                "trust", cfg.TRUST_INITIAL
            ) * decay_factor

    # ---------------------------------------------------------------- #
    #  Mesa step hook
    # ---------------------------------------------------------------- #

    def step(self):
        """Advance the model by one tick."""
        self.schedule.step()

        # NOVEL: passive trust decay after all interactions
        self._decay_trust()

        self.datacollector.collect(self)

        # Stop process if everyone has churned
        if all(not a.active for a in self.schedule.agents):
            self.running = False
