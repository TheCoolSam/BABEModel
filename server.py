"""
server.py — Mesa Visualization Server for the Dynamic-BABE Model
=================================================================
Launches an interactive browser dashboard with:

  1. Network graph  – nodes coloured by mean opinion (blue ↔ red),
     greyed-out when churned.
  2. Line chart     – Global Polarization σ over time.
  3. Line chart     – Platform Revenue over time.
  4. Line chart     – Active Users & Avg Frustration over time.

Run:
    python server.py
Then open http://127.0.0.1:8521 in your browser.
"""

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule, NetworkModule

import config as cfg
from model import SocialNetworkModel


# ====================================================================
# Network portrayal callback
# ====================================================================

def network_portrayal(G):
    """
    Return the portrayal dict consumed by Mesa's NetworkModule.
    Each node is coloured on a blue ↔ red gradient based on the
    agent's mean opinion scalar.  Churned agents are drawn grey.
    """
    portrayal = {"nodes": [], "edges": []}

    for node_id in G.nodes():
        agent = G.nodes[node_id].get("agent")
        if agent is None:
            continue

        if not agent.active:
            colour = "#AAAAAA"        # Churned → grey
            size = 3
        else:
            # Map mean opinion ∈ [-1, 1] → hue  (blue → red)
            val = (float(agent.opinion.mean()) + 1.0) / 2.0  # → [0, 1]
            r = int(val * 255)
            b = int((1.0 - val) * 255)
            colour = f"#{r:02X}00{b:02X}"
            size = 6

        portrayal["nodes"].append(
            {
                "id": node_id,
                "size": size,
                "color": colour,
                "label": None,
            }
        )

    for u, v in G.edges():
        portrayal["edges"].append(
            {"source": u, "target": v, "color": "#CCCCCC", "width": 0.5}
        )

    return portrayal


# ====================================================================
# Visualisation modules
# ====================================================================

network_viz = NetworkModule(network_portrayal, 500, 500)

polarization_chart = ChartModule(
    [{"Label": "Polarization", "Color": "#E74C3C"}],
    data_collector_name="datacollector",
    canvas_height=200,
    canvas_width=500,
)

revenue_chart = ChartModule(
    [{"Label": "Revenue", "Color": "#27AE60"}],
    data_collector_name="datacollector",
    canvas_height=200,
    canvas_width=500,
)

active_frustration_chart = ChartModule(
    [
        {"Label": "Active_Users", "Color": "#2980B9"},
        {"Label": "Avg_Frustration", "Color": "#F39C12"},
    ],
    data_collector_name="datacollector",
    canvas_height=200,
    canvas_width=500,
)

churn_chart = ChartModule(
    [{"Label": "Churn_Rate", "Color": "#8E44AD"}],
    data_collector_name="datacollector",
    canvas_height=200,
    canvas_width=500,
)

# ====================================================================
# Model parameters (adjustable via browser sliders)
# ====================================================================

model_params = {
    "num_agents": cfg.NUM_AGENTS,
    "ba_m": cfg.BA_EDGE_PARAM,
    "enable_bridge": cfg.ENABLE_BRIDGE,
    "seed": cfg.RANDOM_SEED,
}

# ====================================================================
# Server launch
# ====================================================================

server = ModularServer(
    SocialNetworkModel,
    [
        network_viz,
        polarization_chart,
        revenue_chart,
        active_frustration_chart,
        churn_chart,
    ],
    "Dynamic-BABE Model — Polarization Paradox",
    model_params,
)

server.port = 8521

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  Dynamic-BABE Model — Visualisation Server      ║")
    print("║  Open http://127.0.0.1:8521 in your browser     ║")
    print("╚══════════════════════════════════════════════════╝")
    server.launch()
