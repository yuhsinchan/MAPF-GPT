"""
Case 1: Unit test for _propagate_risk.

Tests whether risk propagates transitively along distance-to-goal paths.
No model or C++ module required — pure Python with mock objects.

Setup: 1×7 corridor. Risk source at cell (0,6), goal at (0,0).
Expected (transitive): risk at (0,k) = alpha^(6-k) for k=0..5.
"""

from types import SimpleNamespace

import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gpt.safe_action_wrapper import DecentralizedWrapper


def make_1d_cost2go(length: int, target_y: int) -> list:
    """
    Cost2go for a 1-row corridor: d[0][y] = |y - target_y|.
    Returns a 2D list of shape (1, length).
    """
    return [[abs(y - target_y) for y in range(length)]]


def test_propagate_risk():
    alpha = 0.5
    length = 7
    target = (0, 0)

    cost2go_data = {target: make_1d_cost2go(length, 0)}
    mock_self = SimpleNamespace(cost2go_data=cost2go_data, alpha=alpha)

    risk_map = {(0, 6): 1.0}
    result = DecentralizedWrapper._propagate_risk(mock_self, risk_map, target)

    print("=== Case 1: _propagate_risk on 1×7 corridor ===")
    print(f"Source: (0,6) risk=1.0 | Goal: (0,0) | alpha={alpha}")
    print()

    any_bug = False
    for y in range(length - 1, -1, -1):
        cell = (0, y)
        actual = result.get(cell, 0.0)
        expected = alpha ** (6 - y)
        match = abs(actual - expected) < 1e-6
        marker = "OK" if match else "BUG"
        if not match:
            any_bug = True
        print(f"  (0,{y})  dist_to_source={6-y}  actual={actual:.4f}  expected={expected:.4f}  [{marker}]")

    print()
    cells_with_risk = sorted([c for c, v in result.items() if v > 1e-8], key=lambda c: c[1])
    print(f"Cells with nonzero risk: {cells_with_risk}")

    if any_bug:
        # Check if it's the one-hop bug specifically
        only_one_hop = all(result.get((0, y), 0.0) < 1e-8 for y in range(5))
        if only_one_hop:
            print("\n>>> BUG CONFIRMED: propagation stops after ONE HOP.")
            print("    Only (0,5) received risk from (0,6). Cells (0,0)-(0,4) are zero.")
            print("    The loop iterates over the original risk_map cells only,")
            print("    so newly propagated cells are never re-processed.")
        else:
            print("\n>>> BUG: some cells have wrong values (but propagation went beyond one hop)")
    else:
        print("\n>>> ALL OK: transitive propagation works correctly.")

    return result


def visualize(result, alpha=0.5, length=7):
    actual = [result.get((0, y), 0.0) for y in range(length)]
    expected = [alpha ** (6 - y) for y in range(length)]

    fig, ax = plt.subplots(figsize=(10, 4))
    xs = list(range(length))
    ax.bar([x - 0.18 for x in xs], actual, width=0.34, label="Actual", color="steelblue")
    ax.bar([x + 0.18 for x in xs], expected, width=0.34, label="Expected (transitive)", color="orange", alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"(0,{y})" for y in range(length)])
    ax.set_ylabel("Risk")
    ax.set_title(f"_propagate_risk: 1×7 corridor | source=(0,6) goal=(0,0) α={alpha}")
    ax.legend()

    # Annotate
    ax.annotate("Source\nrisk=1.0", xy=(6, 1.0), xytext=(5.0, 0.85),
                arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax.annotate("Goal", xy=(0, 0), xytext=(0.6, 0.15),
                arrowprops=dict(arrowstyle="->"), fontsize=9)

    out = Path(__file__).parent / "propagation_1d.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"\nSaved: {out}")
    plt.close()


if __name__ == "__main__":
    result = test_propagate_risk()
    visualize(result)
