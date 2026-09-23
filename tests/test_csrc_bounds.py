import ctypes
import numpy as np
import pytest
from chimera.evaluator import (
    ChimeraEngine,
    ChimeraInstruction,
    ChimeraOpcode,
    ChimeraArena,
)

@pytest.fixture
def engine():
    eng = ChimeraEngine(capacity_rows=100)
    yield eng
    eng.close()

def test_csrc_out_reg_bounds_guard(engine):
    """Test that instructions with out_reg >= 16 return -3 safely without memory corruption."""
    ins = ChimeraInstruction(
        op=ChimeraOpcode.ADD,
        out_reg=20,  # Invalid: >= 16
        in_reg1=0,
        in_reg2=1,
        feat_idx=0,
        imm_val=0.0
    )
    ins_array = (ChimeraInstruction * 1)(ins)

    n_rows = 10
    n_cols = 2
    col_ptrs = (ctypes.POINTER(ctypes.c_float) * n_cols)()
    cols = [np.ones(n_rows, dtype=np.float32), np.ones(n_rows, dtype=np.float32)]
    for i, c in enumerate(cols):
        col_ptrs[i] = c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    output = np.empty(n_rows, dtype=np.float32)
    out_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    rc = engine._lib.chimera_execute(
        ins_array,
        ctypes.c_size_t(1),
        col_ptrs,
        ctypes.c_size_t(n_cols),
        ctypes.c_size_t(n_rows),
        ctypes.byref(engine._arena),
        out_ptr
    )
    # Must return -3 (register out of bounds)
    assert rc == -3
    assert output.shape == (n_rows,)

def test_csrc_in_reg_bounds_guard(engine):
    """Test that instructions with in_reg1 or in_reg2 >= 16 return -3 safely."""
    for in1, in2 in [(25, 0), (0, 30), (16, 16)]:
        ins = ChimeraInstruction(
            op=ChimeraOpcode.MUL,
            out_reg=2,
            in_reg1=in1,
            in_reg2=in2,
            feat_idx=0,
            imm_val=0.0
        )
        ins_array = (ChimeraInstruction * 1)(ins)
        n_rows = 10
        col_ptrs = (ctypes.POINTER(ctypes.c_float) * 1)()
        col = np.ones(n_rows, dtype=np.float32)
        col_ptrs[0] = col.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output = np.empty(n_rows, dtype=np.float32)

        rc = engine._lib.chimera_execute(
            ins_array,
            ctypes.c_size_t(1),
            col_ptrs,
            ctypes.c_size_t(1),
            ctypes.c_size_t(n_rows),
            ctypes.byref(engine._arena),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        assert rc == -3
        assert output.shape == (n_rows,)

def test_csrc_feat_null_bounds_guard(engine):
    """Test that out-of-bounds feature indices return -5 without null pointer dereference."""
    ins = ChimeraInstruction(
        op=ChimeraOpcode.LOAD_FEAT,
        out_reg=0,
        in_reg1=0,
        in_reg2=0,
        feat_idx=999,  # >= n_cols (1)
        imm_val=0.0
    )
    ins_array = (ChimeraInstruction * 1)(ins)
    n_rows = 10
    col_ptrs = (ctypes.POINTER(ctypes.c_float) * 1)()
    col = np.ones(n_rows, dtype=np.float32)
    col_ptrs[0] = col.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output = np.empty(n_rows, dtype=np.float32)

    rc = engine._lib.chimera_execute(
        ins_array,
        ctypes.c_size_t(1),
        col_ptrs,
        ctypes.c_size_t(1),
        ctypes.c_size_t(n_rows),
        ctypes.byref(engine._arena),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )
    assert rc == -5
    assert output.shape == (n_rows,)

def test_csrc_arena_reinit_no_leak(engine):
    """Test that reinitializing an already-initialized arena frees old memory cleanly."""
    arena = ChimeraArena()
    rc1 = engine._lib.chimera_init_arena(ctypes.byref(arena), ctypes.c_size_t(1000))
    assert rc1 == 0
    assert arena.is_initialized == 1

    # Re-initialize on same struct without closing first
    rc2 = engine._lib.chimera_init_arena(ctypes.byref(arena), ctypes.c_size_t(2000))
    assert rc2 == 0
    assert arena.capacity_rows == 2000

    # Clean up
    engine._lib.chimera_free_arena(ctypes.byref(arena))
    assert arena.is_initialized == 0
