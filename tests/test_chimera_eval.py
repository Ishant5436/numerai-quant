import numpy as np
import pytest
from chimera.evaluator import ChimeraEngine, InstructionBuilder

@pytest.fixture(scope="module")
def engine():
    eng = ChimeraEngine(capacity_rows=100000)
    yield eng
    eng.close()

def test_arena_lifecycle():
    eng = ChimeraEngine(capacity_rows=50000)
    assert eng.is_initialized
    eng.close()
    assert not eng.is_initialized

def test_basic_arithmetic_parity(engine):
    n = 50000
    np.random.seed(42)
    col0 = np.random.randn(n).astype(np.float32)
    col1 = np.random.randn(n).astype(np.float32)
    features = np.ascontiguousarray(np.column_stack([col0, col1]))

    # Test ADD: feat0 + feat1
    ib = InstructionBuilder()
    ib.load_feat(reg=0, feat_idx=0)
    ib.load_feat(reg=1, feat_idx=1)
    ib.add(out_reg=2, in_reg1=0, in_reg2=1)
    
    out = engine.execute(ib.build(), features)
    expected = col0 + col1
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

    # Test SUB: feat0 - feat1
    ib = InstructionBuilder()
    ib.load_feat(reg=0, feat_idx=0)
    ib.load_feat(reg=1, feat_idx=1)
    ib.sub(out_reg=2, in_reg1=0, in_reg2=1)
    out = engine.execute(ib.build(), features)
    np.testing.assert_allclose(out, col0 - col1, rtol=1e-5, atol=1e-5)

    # Test MUL: feat0 * feat1
    ib = InstructionBuilder()
    ib.load_feat(reg=0, feat_idx=0)
    ib.load_feat(reg=1, feat_idx=1)
    ib.mul(out_reg=2, in_reg1=0, in_reg2=1)
    out = engine.execute(ib.build(), features)
    np.testing.assert_allclose(out, col0 * col1, rtol=1e-5, atol=1e-5)

def test_safe_div_zero_resilience(engine):
    col0 = np.array([1.0, 5.0, -2.0, 0.0], dtype=np.float32)
    col1 = np.array([2.0, 0.0, 0.0, 4.0], dtype=np.float32)
    features = np.ascontiguousarray(np.column_stack([col0, col1]))

    ib = InstructionBuilder()
    ib.load_feat(reg=0, feat_idx=0)
    ib.load_feat(reg=1, feat_idx=1)
    ib.safe_div(out_reg=2, in_reg1=0, in_reg2=1)
    out = engine.execute(ib.build(), features)

    expected = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-6)
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()

def test_transcendental_parity(engine):
    n = 20000
    np.random.seed(1337)
    col0 = np.random.randn(n).astype(np.float32)
    features = np.ascontiguousarray(col0.reshape(-1, 1))

    # Tanh
    ib = InstructionBuilder()
    ib.load_feat(reg=0, feat_idx=0)
    ib.tanh(out_reg=1, in_reg1=0)
    out = engine.execute(ib.build(), features)
    np.testing.assert_allclose(out, np.tanh(col0), rtol=1e-5, atol=1e-5)

    # Sigmoid
    ib = InstructionBuilder()
    ib.load_feat(reg=0, feat_idx=0)
    ib.sigmoid(out_reg=1, in_reg1=0)
    out = engine.execute(ib.build(), features)
    expected_sig = 1.0 / (1.0 + np.exp(-col0))
    np.testing.assert_allclose(out, expected_sig, rtol=1e-5, atol=1e-5)

    # Zscore
    ib = InstructionBuilder()
    ib.load_feat(reg=0, feat_idx=0)
    ib.zscore(out_reg=1, in_reg1=0)
    out = engine.execute(ib.build(), features)
    expected_z = (col0 - np.mean(col0)) / np.std(col0)
    np.testing.assert_allclose(out, expected_z, rtol=1e-4, atol=1e-4)

def test_chained_formula_parity(engine):
    n = 10000
    np.random.seed(99)
    col0 = np.random.randn(n).astype(np.float32)
    col1 = np.random.randn(n).astype(np.float32)
    features = np.ascontiguousarray(np.column_stack([col0, col1]))

    # Formula: tanh(col0 + col1) * max(col0, col1)
    ib = InstructionBuilder()
    ib.load_feat(reg=0, feat_idx=0)
    ib.load_feat(reg=1, feat_idx=1)
    ib.add(out_reg=2, in_reg1=0, in_reg2=1)
    ib.tanh(out_reg=3, in_reg1=2)
    ib.max(out_reg=4, in_reg1=0, in_reg2=1)
    ib.mul(out_reg=5, in_reg1=3, in_reg2=4)

    out = engine.execute(ib.build(), features)
    expected = np.tanh(col0 + col1) * np.maximum(col0, col1)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
