#ifndef CHIMERA_EVAL_HPP
#define CHIMERA_EVAL_HPP

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Power of 10 Invariant Limits
constexpr size_t CHIMERA_MAX_ROWS = 1000000;
constexpr size_t CHIMERA_MAX_REGS = 16;
constexpr size_t CHIMERA_MAX_INSTRS = 64;
constexpr size_t CHIMERA_MAX_FEATURES = 2048;

enum ChimeraOpcode : uint8_t {
    CHIMERA_OP_NOP = 0,
    CHIMERA_OP_LOAD_FEAT = 1,
    CHIMERA_OP_LOAD_IMM = 2,
    CHIMERA_OP_ADD = 3,
    CHIMERA_OP_SUB = 4,
    CHIMERA_OP_MUL = 5,
    CHIMERA_OP_SAFE_DIV = 6,
    CHIMERA_OP_MAX = 7,
    CHIMERA_OP_MIN = 8,
    CHIMERA_OP_TANH = 9,
    CHIMERA_OP_SIGMOID = 10,
    CHIMERA_OP_SIGN = 11,
    CHIMERA_OP_ABS = 12,
    CHIMERA_OP_LOG1P = 13,
    CHIMERA_OP_ZSCORE = 14
};

struct ChimeraInstruction {
    uint8_t op;
    uint8_t out_reg;
    uint8_t in_reg1;
    uint8_t in_reg2;
    uint16_t feat_idx;
    float imm_val;
};

struct ChimeraArena {
    float* buffer;
    size_t capacity_rows;
    size_t num_regs;
    int is_initialized;
};

// Interface exports
int chimera_init_arena(ChimeraArena* arena, size_t max_rows);
void chimera_free_arena(ChimeraArena* arena);
int chimera_execute(
    const ChimeraInstruction* instrs,
    size_t num_instrs,
    const float* const* feature_cols,
    size_t num_features,
    size_t num_rows,
    ChimeraArena* arena,
    float* output
);

#ifdef __cplusplus
}
#endif

#endif // CHIMERA_EVAL_HPP
