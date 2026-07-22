"""
config.py - Dynamic-BABE Model Global Configuration

All simulation constants centralised here for reproducibility.
Change values here; never hard-code numbers in agent.py / model.py.
"""

# Population & Network Topology
NUM_AGENTS = 500
BA_EDGE_PARAM = 3
NUM_ISSUES = 2
OPINION_INIT_MODE = "bipolar"

# Agent Cognitive Parameters
BETA_MEAN = 4.0
BETA_STD = 1.5
BETA_MIN = 0.5
INTERACTIONS_PER_STEP = 1

# Social-Judgment Thresholds
ASSIMILATION_THRESHOLD = 0.3

# Toxic Churn Hypothesis
CHURN_THRESHOLD = 15
HEALING_RATE = 1
BRIDGE_HEALING_BONUS = 2

# Revenue Step-Function
BASE_AD_RATE = 1.0
POLARIZATION_CLIFF = 0.6
SAFE_AD_MULTIPLIER = 1.0
UNSAFE_AD_MULTIPLIER = 0.4

# The Bridge Algorithm
ENABLE_BRIDGE = False
BRIDGE_EFFICACY = 0.46

# Dyadic Trust (NOVEL)
# Per-edge trust variable T_ij ∈ [0, 1] that co-evolves with opinions.
# High trust buffers agents against occasional disagreements —
# "friends can argue without unfriending each other."
# NOTE: Currently symmetric (T_ij = T_ji). Future work should explore
#       asymmetric trust where T_ij ≠ T_ji (I may trust you more than
#       you trust me).
ENABLE_TRUST = False
TRUST_INITIAL = 0.5           # T_0: starting trust for all edges
TRUST_INFLUENCE = 0.3         # alpha: how much trust shifts w_ij upward
TRUST_GAIN = 0.02             # delta_+: trust gained per assimilation event
TRUST_LOSS = 0.05             # delta_-: trust lost per backfire event
TRUST_DECAY = 0.001           # lambda: passive trust decay per step (no interaction)

# Echo Chamber Detection
# Measures structural clustering of like-minded agents and trust
# segregation between ingroup vs outgroup pairs.
ENABLE_ECHO_CHAMBER_METRICS = True

# Simulation Control
MAX_STEPS = 200
BATCH_ITERATIONS = 30
RANDOM_SEED = 42
