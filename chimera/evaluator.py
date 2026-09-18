import os
import sys
import ctypes
import numpy as np
from enum import IntEnum
from typing import List, Optional

class ChimeraOpcode(IntEnum):
    NOP = 0
    LOAD_FEAT = 1
    LOAD_IMM = 2
    ADD = 3
    SUB = 4
    MUL = 5
    SAFE_DIV = 6
    MAX = 7
    MIN = 8
    TANH = 9
    SIGMOID = 10
    SIGN = 11
    ABS = 12
    LOG1P = 13
    ZSCORE = 14

class ChimeraInstruction(ctypes.Structure):
    _fields_ = [
        ("op", ctypes.c_uint8),
        ("out_reg", ctypes.c_uint8),
        ("in_reg1", ctypes.c_uint8),
        ("in_reg2", ctypes.c_uint8),
        ("feat_idx", ctypes.c_uint16),
        ("imm_val", ctypes.c_float),
    ]

class ChimeraArena(ctypes.Structure):
    _fields_ = [
        ("buffer", ctypes.POINTER(ctypes.c_float)),
        ("capacity_rows", ctypes.c_size_t),
        ("num_regs", ctypes.c_size_t),
        ("is_initialized", ctypes.c_int),
    ]

class InstructionBuilder:
    def __init__(self):
        self.instructions: List[ChimeraInstruction] = []

    def load_feat(self, reg: int, feat_idx: int) -> "InstructionBuilder":
        ins = ChimeraInstruction(
            op=ChimeraOpcode.LOAD_FEAT,
            out_reg=reg,
            in_reg1=0,
            in_reg2=0,
            feat_idx=feat_idx,
            imm_val=0.0
        )
        self.instructions.append(ins)
        return self

    def load_imm(self, reg: int, val: float) -> "InstructionBuilder":
        ins = ChimeraInstruction(
            op=ChimeraOpcode.LOAD_IMM,
            out_reg=reg,
            in_reg1=0,
            in_reg2=0,
            feat_idx=0,
            imm_val=float(val)
        )
        self.instructions.append(ins)
        return self

    def add(self, out_reg: int, in_reg1: int, in_reg2: int) -> "InstructionBuilder":
        ins = ChimeraInstruction(op=ChimeraOpcode.ADD, out_reg=out_reg, in_reg1=in_reg1, in_reg2=in_reg2, feat_idx=0, imm_val=0.0)
        self.instructions.append(ins)
        return self

    def sub(self, out_reg: int, in_reg1: int, in_reg2: int) -> "InstructionBuilder":
        ins = ChimeraInstruction(op=ChimeraOpcode.SUB, out_reg=out_reg, in_reg1=in_reg1, in_reg2=in_reg2, feat_idx=0, imm_val=0.0)
        self.instructions.append(ins)
        return self

    def mul(self, out_reg: int, in_reg1: int, in_reg2: int) -> "InstructionBuilder":
        ins = ChimeraInstruction(op=ChimeraOpcode.MUL, out_reg=out_reg, in_reg1=in_reg1, in_reg2=in_reg2, feat_idx=0, imm_val=0.0)
        self.instructions.append(ins)
        return self

    def safe_div(self, out_reg: int, in_reg1: int, in_reg2: int) -> "InstructionBuilder":
        ins = ChimeraInstruction(op=ChimeraOpcode.SAFE_DIV, out_reg=out_reg, in_reg1=in_reg1, in_reg2=in_reg2, feat_idx=0, imm_val=0.0)
        self.instructions.append(ins)
        return self

    def max(self, out_reg: int, in_reg1: int, in_reg2: int) -> "InstructionBuilder":
        ins = ChimeraInstruction(op=ChimeraOpcode.MAX, out_reg=out_reg, in_reg1=in_reg1, in_reg2=in_reg2, feat_idx=0, imm_val=0.0)
        self.instructions.append(ins)
        return self

    def min(self, out_reg: int, in_reg1: int, in_reg2: int) -> "InstructionBuilder":
        ins = ChimeraInstruction(op=ChimeraOpcode.MIN, out_reg=out_reg, in_reg1=in_reg1, in_reg2=in_reg2, feat_idx=0, imm_val=0.0)
        self.instructions.append(ins)
        return self

    def tanh(self, out_reg: int, in_reg1: int) -> "InstructionBuilder":
        ins = ChimeraInstruction(op=ChimeraOpcode.TANH, out_reg=out_reg, in_reg1=in_reg1, in_reg2=0, feat_idx=0, imm_val=0.0)
        self.instructions.append(ins)
        return self

    def sigmoid(self, out_reg: int, in_reg1: int) -> "InstructionBuilder":
        ins = ChimeraInstruction(op=ChimeraOpcode.SIGMOID, out_reg=out_reg, in_reg1=in_reg1, in_reg2=0, feat_idx=0, imm_val=0.0)
        self.instructions.append(ins)
        return self

    def sign(self, out_reg: int, in_reg1: int) -> "InstructionBuilder":
        ins = ChimeraInstruction(op=ChimeraOpcode.SIGN, out_reg=out_reg, in_reg1=in_reg1, in_reg2=0, feat_idx=0, imm_val=0.0)
        self.instructions.append(ins)
        return self

    def abs(self, out_reg: int, in_reg1: int) -> "InstructionBuilder":
        ins = ChimeraInstruction(op=ChimeraOpcode.ABS, out_reg=out_reg, in_reg1=in_reg1, in_reg2=0, feat_idx=0, imm_val=0.0)
        self.instructions.append(ins)
        return self

    def log1p(self, out_reg: int, in_reg1: int) -> "InstructionBuilder":
        ins = ChimeraInstruction(op=ChimeraOpcode.LOG1P, out_reg=out_reg, in_reg1=in_reg1, in_reg2=0, feat_idx=0, imm_val=0.0)
        self.instructions.append(ins)
        return self

    def zscore(self, out_reg: int, in_reg1: int) -> "InstructionBuilder":
        ins = ChimeraInstruction(op=ChimeraOpcode.ZSCORE, out_reg=out_reg, in_reg1=in_reg1, in_reg2=0, feat_idx=0, imm_val=0.0)
        self.instructions.append(ins)
        return self

    def build(self) -> List[ChimeraInstruction]:
        return self.instructions

class ChimeraEngine:
    def __init__(self, capacity_rows: int = 100000, lib_path: Optional[str] = None):
        self.is_initialized = False
        assert capacity_rows > 0 and capacity_rows <= 1000000, "Capacity rows must be in (0, 1000000]"
        self.capacity_rows = capacity_rows
        
        if lib_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            ext = ".dylib" if sys.platform == "darwin" else ".dll" if sys.platform == "win32" else ".so"
            candidate = os.path.join(base_dir, "csrc", f"libchimera_eval{ext}")
            if not os.path.exists(candidate):
                for alt_ext in [".so", ".dylib"]:
                    alt_path = os.path.join(base_dir, "csrc", f"libchimera_eval{alt_ext}")
                    if os.path.exists(alt_path):
                        candidate = alt_path
                        break
            lib_path = candidate
        
        assert os.path.exists(lib_path), f"Native library missing: {lib_path}"
        self._lib = ctypes.CDLL(lib_path)
        
        # Function signatures
        self._lib.chimera_init_arena.argtypes = [ctypes.POINTER(ChimeraArena), ctypes.c_size_t]
        self._lib.chimera_init_arena.restype = ctypes.c_int
        
        self._lib.chimera_free_arena.argtypes = [ctypes.POINTER(ChimeraArena)]
        self._lib.chimera_free_arena.restype = None
        
        self._lib.chimera_execute.argtypes = [
            ctypes.POINTER(ChimeraInstruction),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ChimeraArena),
            ctypes.POINTER(ctypes.c_float)
        ]
        self._lib.chimera_execute.restype = ctypes.c_int

        self._arena = ChimeraArena()
        rc = self._lib.chimera_init_arena(ctypes.byref(self._arena), ctypes.c_size_t(capacity_rows))
        assert rc == 0, f"Failed to initialize Chimera arena, error code: {rc}"
        self.is_initialized = True

    def close(self):
        if getattr(self, "is_initialized", False):
            self._lib.chimera_free_arena(ctypes.byref(self._arena))
            self.is_initialized = False

    def __del__(self):
        self.close()

    def execute(self, instructions: List[ChimeraInstruction], features: np.ndarray) -> np.ndarray:
        assert self.is_initialized, "Engine arena is not initialized"
        assert len(instructions) > 0, "Instruction list cannot be empty"
        assert isinstance(features, np.ndarray), "Features must be a numpy ndarray"
        assert features.dtype == np.float32, "Features array must be float32"
        
        n_rows, n_cols = features.shape
        assert n_rows <= self.capacity_rows, f"Rows ({n_rows}) exceed arena capacity ({self.capacity_rows})"
        
        # Prepare instruction array
        ins_array = (ChimeraInstruction * len(instructions))(*instructions)
        
        # Prepare pointers to columns
        # To avoid copies, features must be Fortran/column contiguous or we extract column pointers
        # A simple zero-copy approach: ensure each column is contiguous
        col_pointers = (ctypes.POINTER(ctypes.c_float) * n_cols)()
        col_holders = []
        for c in range(n_cols):
            col = np.ascontiguousarray(features[:, c], dtype=np.float32)
            col_holders.append(col)
            col_pointers[c] = col.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
        output = np.empty(n_rows, dtype=np.float32)
        out_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        rc = self._lib.chimera_execute(
            ins_array,
            ctypes.c_size_t(len(instructions)),
            col_pointers,
            ctypes.c_size_t(n_cols),
            ctypes.c_size_t(n_rows),
            ctypes.byref(self._arena),
            out_ptr
        )
        assert rc == 0, f"chimera_execute returned non-zero error: {rc}"
        return output
