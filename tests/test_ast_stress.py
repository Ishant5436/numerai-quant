import pytest
import numpy as np
from chimera.ast_generator import (
    FeatureNode,
    ImmNode,
    UnaryNode,
    BinaryNode,
    ChimeraOpcode,
    ASTGenerator,
)
from chimera.evaluator import ChimeraEngine

@pytest.fixture(scope="module")
def engine():
    eng = ChimeraEngine(capacity_rows=500)
    yield eng
    eng.close()

def test_deep_right_leaning_tree_register_safety(engine):
    """Stress test right-leaning tree where right subtree emits many instructions.
    
    Verifies that left operand register is NEVER clobbered by right subtree
    evaluation, directly proving fix for Bug C2.
    """
    n_levels = 10
    # Build: feat(0) + (feat(1) + (feat(2) + ... + feat(10)))
    curr = FeatureNode(feat_idx=n_levels)
    for i in range(n_levels - 1, -1, -1):
        curr = BinaryNode(op=ChimeraOpcode.ADD, left=FeatureNode(feat_idx=i), right=curr)

    assert curr.depth() == n_levels + 1
    instructions = curr.compile_to_bytecode()
    assert len(instructions) >= n_levels

    # Verify all instruction registers remain within [0, 15]
    for ins in instructions:
        assert ins.out_reg < 16
        assert ins.in_reg1 < 16
        assert ins.in_reg2 < 16

    # Verify execution against ground truth
    n_rows = 50
    features = np.ones((n_rows, n_levels + 1), dtype=np.float32)
    output = engine.execute(instructions, features)
    
    # Each row is sum of (n_levels + 1) ones = 11.0
    expected = float(n_levels + 1)
    assert np.allclose(output, expected, atol=1e-4)
    assert not np.isnan(output).any()

def test_deep_unary_chain_inplace_register_safety(engine):
    """Stress test long chain of unary operators to verify in-place register reuse."""
    n_ops = 25
    curr = FeatureNode(feat_idx=0)
    for _ in range(n_ops):
        curr = UnaryNode(op=ChimeraOpcode.TANH, child=curr)

    assert curr.depth() == n_ops + 1
    instructions = curr.compile_to_bytecode()
    assert len(instructions) == n_ops + 1

    # Verify all registers are valid
    for ins in instructions:
        assert ins.out_reg < 16
        assert ins.in_reg1 < 16

    n_rows = 30
    features = np.full((n_rows, 1), 0.5, dtype=np.float32)
    output = engine.execute(instructions, features)

    # Compute Python ground truth
    val = 0.5
    for _ in range(n_ops):
        val = np.tanh(val)

    assert np.allclose(output, val, atol=1e-4)
    assert output.shape == (n_rows,)

def test_balanced_binary_tree_register_recycling(engine):
    """Stress test balanced tree to verify registers are recycled back to free pool."""
    def build_balanced(depth: int, feat_offset: int):
        assert depth >= 0
        if depth == 0:
            return FeatureNode(feat_idx=feat_offset)
        left = build_balanced(depth - 1, feat_offset)
        right = build_balanced(depth - 1, feat_offset + 1)
        return BinaryNode(op=ChimeraOpcode.ADD, left=left, right=right)

    tree = build_balanced(depth=4, feat_offset=0)
    assert tree.depth() == 5
    instructions = tree.compile_to_bytecode()
    assert len(instructions) > 15

    for ins in instructions:
        assert ins.out_reg < 16
        assert ins.in_reg1 < 16

    n_rows = 20
    features = np.ones((n_rows, 16), dtype=np.float32)
    output = engine.execute(instructions, features)
    assert not np.isnan(output).any()
    assert output.shape == (n_rows,)
