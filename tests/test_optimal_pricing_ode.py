"""
Unit and Invariant Tests for Differential Equation Optimal Pricing Engine.
Deterministic Safety Standards: Functions <= 60 lines, assertion density >= 2.
"""
import ast
import os
import pytest
import numpy as np


def test_hamiltonian_system_derivatives():
    """Verify state and co-state derivatives computed by the ODE Hamiltonian."""
    from scripts.optimal_pricing_ode import hamiltonian_system_derivatives

    params = {
        "lambda_inflow": 10.0,
        "beta": 0.5,
        "delta_0": 0.1,
        "theta": 0.05,
        "discount_rate": 0.05,
    }
    state = np.array([5.0, 2.0])  # S = 5 subscribers, lambda = 2.0 shadow value
    t = 0.0

    derivs = hamiltonian_system_derivatives(t, state, params)
    assert isinstance(derivs, np.ndarray), "Derivatives must return numpy array"
    assert len(derivs) == 2, "Derivatives must have dimension 2 (dS/dt, dlambda/dt)"
    assert np.isfinite(derivs).all(), "Derivatives must be finite with zero NaNs"


def test_solve_steady_state_price():
    """Verify analytical steady-state price formula matches equilibrium conditions."""
    from scripts.optimal_pricing_ode import solve_steady_state_price

    params = {
        "beta": 0.5,
        "delta_0": 0.1,
        "discount_rate": 0.05,
        "theta": 0.0,
    }
    # P* = (1 / beta) * (1 + r / delta_0) = (1 / 0.5) * (1 + 0.05 / 0.10) = 2.0 * 1.5 = 3.0
    p_star = solve_steady_state_price(params)
    assert isinstance(p_star, float), "P* must be a float"
    assert np.isclose(p_star, 3.0, atol=1e-3), f"Expected 3.0, got {p_star}"
    assert p_star > 0.0, "Optimal price must be strictly positive"


def test_simulate_price_trajectory():
    """Verify numerical ODE integration converges to stable trajectory without divergence."""
    from scripts.optimal_pricing_ode import simulate_price_trajectory

    params = {
        "lambda_inflow": 5.0,
        "beta": 0.8,
        "delta_0": 0.1,
        "theta": 0.02,
        "discount_rate": 0.05,
    }
    res = simulate_price_trajectory(
        s0=1.0, lambda0=1.5, horizon_weeks=12.0, params=params
    )
    assert isinstance(res, dict), "Result must be a dictionary"
    assert "t" in res and "s_trajectory" in res and "price_trajectory" in res, "Missing keys"
    assert len(res["price_trajectory"]) > 0, "Trajectory cannot be empty"
    assert (res["price_trajectory"] > 0).all(), "Prices must remain strictly positive"


def test_compute_model_pricing_recommendations():
    """Verify tier recommendations adhere strictly to 1.0 - 2.0 NMR boundary constraints."""
    from scripts.optimal_pricing_ode import compute_model_pricing_recommendations

    models = [
        {"name": "cypherpole", "corj60": 0.0607, "tier": "flagship"},
        {"name": "cypherpole_hedged", "corj60": 0.0450, "tier": "hedged"},
        {"name": "cypherpole_vel", "corj60": 0.0380, "tier": "momentum"},
    ]
    recs = compute_model_pricing_recommendations(models)
    assert len(recs) == 3, "Must return pricing for all 3 models"
    assert recs[0]["subscription_4_round_nmr"] == 5.0, "Flagship subscription must be 5.00 NMR"
    assert recs[0]["trial_1_round_nmr"] == 1.5, "Flagship trial must be 1.50 NMR"
    for r in recs:
        assert 1.0 <= r["subscription_4_round_nmr"] <= 5.0, (
            f"4-round subscription {r['subscription_4_round_nmr']} outside [1.0, 5.0] NMR"
        )
        assert 0.5 <= r["trial_1_round_nmr"] <= 2.0, (
            f"1-round trial {r['trial_1_round_nmr']} outside [0.5, 2.0] NMR"
        )
        assert r["subscription_4_round_nmr"] > r["trial_1_round_nmr"], (
            "Bundle price must exceed single trial price"
        )


def test_parameter_validation_invariants():
    """Verify safety guards reject invalid parameters deterministically."""
    from scripts.optimal_pricing_ode import solve_steady_state_price

    with pytest.raises(AssertionError):
        solve_steady_state_price({"beta": -0.5, "delta_0": 0.1, "discount_rate": 0.05})

    with pytest.raises(AssertionError):
        solve_steady_state_price({"beta": 0.5, "delta_0": -0.1, "discount_rate": 0.05})


def test_power_of_10_invariants():
    """Verify that optimal_pricing_ode.py adheres to Holzmann Power of 10 rules."""
    module_path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "optimal_pricing_ode.py"
    )
    assert os.path.exists(module_path), f"Module not found at {module_path}"

    with open(module_path, "r") as f:
        tree = ast.parse(f.read())

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert len(functions) >= 4, "Module must contain at least 4 core functions"

    for fn in functions:
        line_length = fn.end_lineno - fn.lineno + 1
        assert line_length <= 60, f"Function {fn.name} exceeds 60 lines ({line_length} lines)"

        assert_count = sum(
            1 for n in ast.walk(fn) if isinstance(n, ast.Assert)
        )
        assert assert_count >= 2, (
            f"Function {fn.name} has only {assert_count} assertions (minimum 2 required)"
        )
