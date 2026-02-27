"""
agent.py — SocialAgent for the Dynamic-BABE Model
===================================================
Implements the full state-vector architecture described in the
theoretical specification:

  • N-dimensional Opinion  (Ō_i ∈ [-1, 1]^N)
  • Salience weighting      (S̄_i, Σ = 1)
  • Cognitive Entrenchment  (β_i ≥ 0)          ← Chen et al. (2021)
  • Behavioral Frustration  (F_i ∈ ℤ≥0)       ← NOVEL
  • Active / Churned status                     ← NOVEL

All heavy math uses numpy for O(n) vectorised performance.
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

        # ── Opinion vector  Ō_i ∈ [-1, 1]^N ────────────────────────
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

        # ── Salience vector  S̄_i  (Dirichlet → sums to 1) ───────────
        self.salience: np.ndarray = rng.dirichlet(np.ones(n))

        # ── Cognitive entrenchment  β_i ~ N(μ, σ²), floored ──────────
        #    Source: Chen et al. (2021) BEBA framework
        self.beta: float = max(
            cfg.BETA_MIN,
            rng.normal(cfg.BETA_MEAN, cfg.BETA_STD),
        )

        # ── NOVEL: Behavioral frustration counter ─────────────────────
        self.frustration: int = 0

        # ── NOVEL: Churn flag ─────────────────────────────────────────
        self.active: bool = True

    # ------------------------------------------------------------------ #
    #  Core maths (called by model.py interaction loop)
    # ------------------------------------------------------------------ #

    def compute_alignment(self, other):
        """
        Weighted dot-product alignment  A_ij.

            A_ij = (Ō_i · Ō_j) × mean(S̄_i, S̄_j)

        Returns a scalar in approximately [-1, 1].
        """
        avg_salience = (self.salience + other.salience) / 2.0
        return float(np.dot(self.opinion * avg_salience, other.opinion))

    def compute_influence_weight(self, alignment):
        """
        Chen et al. (2021) influence weight:

            w_ij = 1 + (β_i × A_ij)

        The '1' encodes baseline trust; high β with negative alignment
        can drive w_ij well below zero (backfire territory).
        """
        return 1.0 + (self.beta * alignment)

    # ------------------------------------------------------------------ #
    #  Jager-Amblard Social Judgment Zones
    # ------------------------------------------------------------------ #

    @staticmethod
    def classify_zone(w_ij):
        """
        Map influence weight to a Social Judgment Theory zone.

        Zone 1 — Latitude of Acceptance   (w > 0.3)   → Assimilation
        Zone 2 — Latitude of Non-Commit.  (0 ≤ w ≤ 0.3) → Ignore
        Zone 3 — Latitude of Rejection    (w < 0)     → Backfire
        """
        if w_ij > cfg.ASSIMILATION_THRESHOLD:
            return 1
        elif w_ij >= 0.0:
            return 2
        else:
            return 3

    # ------------------------------------------------------------------ #
    #  Update rules
    # ------------------------------------------------------------------ #

    def assimilate(self, other, w_ij):
        """
        Modified DeGroot (1974) consensus update — CONSTRUCTIVE path,
        with confirmation-bias dampening.

        Step size μ is attenuated by cognitive entrenchment:
            μ = w_ij / (1 + β_i)      ← high β resists change
            Ō_i(t+1) = Ō_i(t) + μ · (Ō_j(t) − Ō_i(t))

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
        Repulsion path — BACKFIRE.

        Agent i's opinion is pushed *away* from agent j:
            repulsion = |w_ij| / (1 + β_i)
            Ō_i(t+1) = Ō_i(t) + repulsion · (Ō_i(t) − Ō_j(t))

        This moves agent i in the OPPOSITE direction from j,
        amplifying polarization. The step is dampened by β so
        already-extreme agents don't overshoot.

        ── NOVEL CONTRIBUTION ──────────────────────────────────────
        Each backfire event increments F_i (Behavioral Frustration).
        """
        # Push opinion AWAY from the other agent
        repulsion = abs(w_ij) / (1.0 + self.beta)
        away = self.opinion - other.opinion      # Direction away from j
        self.opinion = np.clip(self.opinion + repulsion * away, -1.0, 1.0)

        # ── NOVEL: frustration accrual ────────────────────────────
        self.frustration += 1

    # ------------------------------------------------------------------ #
    #  NOVEL: Toxic Churn Check
    # ------------------------------------------------------------------ #

    def check_churn(self):
        """
        If cumulative frustration exceeds T_c, the agent churns.
        Once churned the agent is permanently inactive and ceases
        all future interactions and revenue generation.
        """
        if self.frustration > cfg.CHURN_THRESHOLD:
            self.active = False

    # ------------------------------------------------------------------ #
    #  Mesa scheduler hook
    # ------------------------------------------------------------------ #

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
