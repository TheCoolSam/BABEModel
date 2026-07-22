"""
agent.py — SocialAgent for the Dynamic-BABE Model
===================================================
Implements the full state-vector architecture described in the
theoretical specification:

  * N-dimensional Opinion
  * Salience weighting
  * Cognitive Entrenchment <- Chen et al. (2021)
  * Behavioral Frustration <- NOVEL
  * Active / Churned status <- NOVEL

All heavy math uses numpy.
"""

import numpy as np
from mesa import Agent

import config as cfg


class SocialAgent(Agent):
    """An opinion-bearing agent on a scale-free social network."""

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def __init__(self, unique_id, model, rng):
        super().__init__(unique_id, model)

        n = cfg.NUM_ISSUES

        # Opinion vector
        if getattr(cfg, 'OPINION_INIT_MODE', 'uniform') == 'bipolar':
            # Bipolar: 50/50 split into two camps (±0.7 ± noise)
            # Models an already-polarised society (realistic for social media)
            pole = 1.0 if rng.random() < 0.5 else -1.0
            self.opinion = np.clip(
                pole * rng.uniform(0.4, 1.0, size=n), -1.0, 1.0
            )
        else:
            # Uniform: classic U(-1, 1) initialization
            self.opinion = rng.uniform(-1.0, 1.0, size=n)

        # Salience vector (Dirichlet sums to 1)
        self.salience: np.ndarray = rng.dirichlet(np.ones(n))

        # Cognitive entrenchment, normally distributed and floored
        #    Source: Chen et al. (2021) BEBA framework
        self.beta: float = max(
            cfg.BETA_MIN,
            rng.normal(cfg.BETA_MEAN, cfg.BETA_STD),
        )

        # NOVEL: Behavioral frustration counter
        self.frustration: int = 0

        # NOVEL: Churn flag
        self.active: bool = True

    # ------------------------------------------------------------------ #
    #  Trust accessor (NOVEL — Dyadic Trust extension)
    # ------------------------------------------------------------------ #

    def get_trust(self, other):
        """
        Query the symmetric trust weight on the edge (self, other).

        Trust is stored on the network edge, not on the agent, because
        it is a dyadic (relationship) property.
        Returns cfg.TRUST_INITIAL if trust is disabled or edge is missing.

        NOTE: Currently symmetric (T_ij = T_ji). Future work should
              explore asymmetric trust.
        """
        if not self.model.enable_trust:
            return cfg.TRUST_INITIAL
        # Symmetric by design: T_ij = T_ji (stored once per undirected edge).
        G = self.model.G
        if G.has_edge(self.unique_id, other.unique_id):
            return G[self.unique_id][other.unique_id].get(
                'trust', cfg.TRUST_INITIAL
            )
        return cfg.TRUST_INITIAL

    # ------------------------------------------------------------------ #
    #  Core maths (called by model.py interaction loop)
    # ------------------------------------------------------------------ #

    def compute_alignment(self, other):
        """
        Weighted dot-product alignment.

        Returns a scalar in approximately [-1, 1].
        """
        avg_salience = (self.salience + other.salience) / 2.0
        return float(np.dot(self.opinion * avg_salience, other.opinion))

    def compute_influence_weight(self, alignment):
        """
        Chen et al. (2021) influence weight

        The '1' encodes baseline trust; high beta with negative alignment
        can drive w_ij well below zero (backfire territory).
        """
        return 1.0 + (self.beta * alignment)

    @staticmethod
    def classify_zone(w_ij):
        """
        Map influence weight to a Social Judgment Theory zone.

        Zone 1 - Latitude of Acceptance -> Assimilation
        Zone 2 - Latitude of Non-Commitment -> Ignore
        Zone 3 - Latitude of Rejection -> Backfire
        """
        if w_ij > cfg.ASSIMILATION_THRESHOLD:
            return 1
        elif w_ij >= 0.0:
            return 2
        else:
            return 3

    def assimilate(self, other, w_ij):
        """
        Modified DeGroot (1974) consensus update - CONSTRUCTIVE path,
        with confirmation-bias dampening.

        Step size is attenuated by cognitive entrenchment: high beta resists change.

        This preserves DeGroot's direction (pull toward the other)
        but entrenched agents move much less per interaction.
        """
        mu = w_ij / (1.0 + self.beta)    # Dampened step size
        delta = other.opinion - self.opinion
        self.opinion = np.clip(self.opinion + mu * delta, -1.0, 1.0)

        # Healing: reduce accumulated frustration
        self.frustration = max(0, self.frustration - cfg.HEALING_RATE)

    def backfire(self, other, w_ij):
        """
        Repulsion path - BACKFIRE.

        Agent's opinion is pushed *away* from the other agent.

        This moves agent i in the OPPOSITE direction from j,
        amplifying polarization. The step is dampened by beta so
        already-extreme agents don't overshoot.

        # NOVEL CONTRIBUTION
        Each backfire event increments F_i (Behavioral Frustration).
        """
        # Push opinion AWAY from the other agent
        repulsion = abs(w_ij) / (1.0 + self.beta)
        away = self.opinion - other.opinion      # Direction away from j
        self.opinion = np.clip(self.opinion + repulsion * away, -1.0, 1.0)

        # NOVEL: frustration accrual
        self.frustration += 1

    def check_churn(self):
        """
        If cumulative frustration exceeds T_c, the agent churns.
        Once churned the agent is permanently inactive and ceases
        all future interactions and revenue generation.
        """
        if self.frustration > cfg.CHURN_THRESHOLD:
            self.active = False

    def step(self):
        """
        Called by the Mesa scheduler each tick.
        Inactive (churned) agents do nothing.
        Interaction partner selection is handled by the Model to keep
        the network topology logic centralised.
        """
        if not self.active:
            return
        # Actual interaction dispatched from model.py → interact()
        self.model.interact(self)
