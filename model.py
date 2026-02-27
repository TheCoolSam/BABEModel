"""
model.py — SocialNetworkModel for the Dynamic-BABE ABM

Implementations of the full model:
  • Barabasi–Albert network generation
  • Per-tick agent interaction loop
  • Bridge Algorithm interception (toggle)
  • Revenue step-function computation
  • Mesa DataCollector for all four KPIs

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
    Global polarization σ — root-mean-square of per-dimension
    standard deviations across all active opinion vectors.

    This avoids the "Data Flattening" trap: np.mean(opinion)
    would cancel out multidimensional extremism (e.g. [+1,-1]
    looks like 0).  Instead we measure spread in EACH dimension
    and combine via RMS.

        σ = sqrt( mean( std(dim_0)², std(dim_1)², ... ) )
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
      Base = Σ(Active Users) × Base Ad Rate
      If σ ≥ 0.6 → Ad Rate drops to 40 % (Brand Safety Penalty).
    """
    active_count = len(_active_agents(model))
    sigma = compute_polarization(model)
    multiplier = (
        cfg.SAFE_AD_MULTIPLIER
        if sigma < cfg.POLARIZATION_CLIFF
        else cfg.UNSAFE_AD_MULTIPLIER
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
# Model
# ====================================================================

class SocialNetworkModel(Model):
    """
    Dynamic-BABE (Biased Assimilation & Behavioral Entrenchment) Model.

    Parameters
    ----------
    num_agents      : population size  (default from config)
    ba_m            : Barabasi–Albert edge parameter
    enable_bridge   : activate the Bridge Algorithm dampener
    seed            : master RNG seed for full reproducibility
    """

    def __init__(
        self,
        num_agents=cfg.NUM_AGENTS,
        ba_m=cfg.BA_EDGE_PARAM,
        enable_bridge=cfg.ENABLE_BRIDGE,
        seed=cfg.RANDOM_SEED,
    ):
        super().__init__()

        # ── Reproducibility ───────────────────────────────────────
        self.rng = np.random.default_rng(seed)
        self._seed = seed

        # ── Parameters ────────────────────────────────────────────
        self.num_agents = num_agents
        self.enable_bridge = enable_bridge

        # ── Network topology: Barabási–Albert scale-free graph ────
        self.G = nx.barabasi_albert_graph(
            n=num_agents, m=ba_m, seed=int(self.rng.integers(0, 2**31))
        )

        # ── Scheduler ─────────────────────────────────────────────
        self.schedule = RandomActivation(self)

        # ── Populate ──────────────────────────────────────────────
        for node_id in self.G.nodes():
            agent = SocialAgent(node_id, self, self.rng)
            self.schedule.add(agent)
            self.G.nodes[node_id]["agent"] = agent

        # ── Data collection ───────────────────────────────────────
        self.datacollector = DataCollector(
            model_reporters={
                "Polarization": compute_polarization,
                "Revenue": compute_revenue,
                "Churn_Rate": compute_churn_rate,
                "Avg_Frustration": compute_avg_frustration,
                "Active_Users": compute_active_users,
            },
            agent_reporters={
                # Extremity = ||O_i|| / sqrt(N)  —  captures multidimensional
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

        # ── Bookkeeping ──────────────────────────────────────────
        self.running = True
        self.datacollector.collect(self)  # Collect initial state (t = 0)

    # ---------------------------------------------------------------- #
    #  Interaction dispatcher  (called from SocialAgent.step)
    # ---------------------------------------------------------------- #

    def interact(self, agent):
        """
        Select a random active neighbour and run the full
        Chen → Jager-Amblard → DeGroot pipeline.
        """
        neighbours = list(self.G.neighbors(agent.unique_id))
        if not neighbours:
            return

        # Filter to active neighbours only
        active_neighbours = [
            self.G.nodes[nid]["agent"]
            for nid in neighbours
            if self.G.nodes[nid]["agent"].active
        ]
        if not active_neighbours:
            return

        # Sample up to INTERACTIONS_PER_STEP neighbours
        k = min(cfg.INTERACTIONS_PER_STEP, len(active_neighbours))
        partners = self.rng.choice(active_neighbours, size=k, replace=False)

        for partner in partners:
            self._process_interaction(agent, partner)

    def _process_interaction(self, agent_i, agent_j):
        """
        Full interaction pipeline for agent_i receiving influence
        from agent_j.

        Step 1  — Alignment           A_ij  (weighted dot-product)
        Step 2  — Influence weight     w_ij  (Chen et al., 2021)
        Step 3  — Social Judgment zone mapping (Jager-Amblard)
        Step 4  — Update / Backfire    (Modified DeGroot)
        Step 5  — Churn check          (NOVEL)
        """
        # ── Step 1: Alignment ─────────────────────────────────────
        alignment = agent_i.compute_alignment(agent_j)

        # ── Step 2: Influence weight  (Chen 2021) ─────────────────
        w_ij = agent_i.compute_influence_weight(alignment)

        # ── Step 3: Zone classification  (Jager-Amblard) ──────────
        zone = SocialAgent.classify_zone(w_ij)

        # ────────────────────────────────────────────────────────
        #  NOVEL CONTRIBUTION — Bridge Algorithm Interception
        # ────────────────────────────────────────────────────────
        if self.enable_bridge and zone == 3:
            # Probabilistic dampener — 46 % 
            if self.rng.random() < cfg.BRIDGE_EFFICACY:
                # SUCCESS: neutralise the backfire → Zone 2
                w_ij = 0.0
                zone = 2  # Non-commitment (frustration NOT incremented)
                # NOVEL: successful moderation actively restores trust
                agent_i.frustration = max(
                    0, agent_i.frustration - cfg.BRIDGE_HEALING_BONUS
                )

        # ── Step 4: Apply update rule ─────────────────────────────
        if zone == 1:
            # Latitude of Acceptance → Assimilation (DeGroot)
            agent_i.assimilate(agent_j, w_ij)
        elif zone == 2:
            # Latitude of Non-Commitment → Ignore (no opinion change)
            pass
        elif zone == 3:
            # Latitude of Rejection → Backfire (repulsion + F_i++)
            agent_i.backfire(agent_j, w_ij)

        # ── Step 5: NOVEL — Toxic Churn check ─────────────────────
        agent_i.check_churn()

    # ---------------------------------------------------------------- #
    #  Mesa step hook
    # ---------------------------------------------------------------- #

    def step(self):
        """Advance the model by one tick."""
        self.schedule.step()
        self.datacollector.collect(self)

        # Stop process if everyone has churned
        if all(not a.active for a in self.schedule.agents):
            self.running = False
