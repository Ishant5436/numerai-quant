#include "chimera_eval.hpp"
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <cstring>

int chimera_init_arena(ChimeraArena* arena, size_t max_rows) {
    assert(arena != nullptr);
    assert(max_rows > 0 && max_rows <= CHIMERA_MAX_ROWS);
    
    if (arena == nullptr || max_rows == 0 || max_rows > CHIMERA_MAX_ROWS) {
        return -1;
    }
    const size_t total_elements = CHIMERA_MAX_REGS * max_rows;
    arena->buffer = static_cast<float*>(std::calloc(total_elements, sizeof(float)));
    if (arena->buffer == nullptr) {
        return -2;
    }
    arena->capacity_rows = max_rows;
    arena->num_regs = CHIMERA_MAX_REGS;
    arena->is_initialized = 1;
    return 0;
}

void chimera_free_arena(ChimeraArena* arena) {
    assert(arena != nullptr);
    assert(arena->is_initialized == 1 || arena->is_initialized == 0);
    if (arena != nullptr && arena->buffer != nullptr) {
        std::free(arena->buffer);
        arena->buffer = nullptr;
        arena->capacity_rows = 0;
        arena->num_regs = 0;
        arena->is_initialized = 0;
    }
}

static int eval_load(float* out, const float* src, float imm, uint8_t op, size_t n) {
    assert(out != nullptr);
    assert(n <= CHIMERA_MAX_ROWS);
    if (op == CHIMERA_OP_LOAD_FEAT) {
        assert(src != nullptr);
        std::memcpy(out, src, n * sizeof(float));
    } else if (op == CHIMERA_OP_LOAD_IMM) {
        for (size_t i = 0; i < n && i < CHIMERA_MAX_ROWS; ++i) {
            out[i] = imm;
        }
    } else {
        return -1;
    }
    return 0;
}

static int eval_arith(float* out, const float* r1, const float* r2, uint8_t op, size_t n) {
    assert(out != nullptr && r1 != nullptr);
    assert(r2 != nullptr && n <= CHIMERA_MAX_ROWS);
    for (size_t i = 0; i < n && i < CHIMERA_MAX_ROWS; ++i) {
        const float v1 = r1[i];
        const float v2 = r2[i];
        if (op == CHIMERA_OP_ADD) {
            out[i] = v1 + v2;
        } else if (op == CHIMERA_OP_SUB) {
            out[i] = v1 - v2;
        } else if (op == CHIMERA_OP_MUL) {
            out[i] = v1 * v2;
        } else if (op == CHIMERA_OP_SAFE_DIV) {
            out[i] = (std::fabs(v2) > 1e-8f) ? (v1 / v2) : 0.0f;
        } else {
            return -1;
        }
    }
    return 0;
}

static int eval_extrema(float* out, const float* r1, const float* r2, uint8_t op, size_t n) {
    assert(out != nullptr && r1 != nullptr);
    assert(r2 != nullptr && n <= CHIMERA_MAX_ROWS);
    for (size_t i = 0; i < n && i < CHIMERA_MAX_ROWS; ++i) {
        const float v1 = r1[i];
        const float v2 = r2[i];
        if (op == CHIMERA_OP_MAX) {
            out[i] = (v1 > v2) ? v1 : v2;
        } else if (op == CHIMERA_OP_MIN) {
            out[i] = (v1 < v2) ? v1 : v2;
        } else {
            return -1;
        }
    }
    return 0;
}

static int eval_unary(float* out, const float* r1, uint8_t op, size_t n) {
    assert(out != nullptr && r1 != nullptr);
    assert(n <= CHIMERA_MAX_ROWS);
    for (size_t i = 0; i < n && i < CHIMERA_MAX_ROWS; ++i) {
        const float v = r1[i];
        if (op == CHIMERA_OP_TANH) {
            out[i] = std::tanh(v);
        } else if (op == CHIMERA_OP_SIGMOID) {
            out[i] = 1.0f / (1.0f + std::exp(-v));
        } else if (op == CHIMERA_OP_SIGN) {
            out[i] = (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f);
        } else if (op == CHIMERA_OP_ABS) {
            out[i] = std::fabs(v);
        } else if (op == CHIMERA_OP_LOG1P) {
            out[i] = std::log1p(std::fabs(v));
        } else {
            return -1;
        }
    }
    return 0;
}

static int eval_zscore(float* out, const float* r1, size_t n) {
    assert(out != nullptr && r1 != nullptr);
    assert(n > 0 && n <= CHIMERA_MAX_ROWS);
    double sum = 0.0;
    for (size_t i = 0; i < n && i < CHIMERA_MAX_ROWS; ++i) {
        sum += static_cast<double>(r1[i]);
    }
    const float mean = static_cast<float>(sum / static_cast<double>(n));
    double sq_diff = 0.0;
    for (size_t i = 0; i < n && i < CHIMERA_MAX_ROWS; ++i) {
        const double d = static_cast<double>(r1[i]) - static_cast<double>(mean);
        sq_diff += d * d;
    }
    const float std_dev = static_cast<float>(std::sqrt(sq_diff / static_cast<double>(n)));
    const float denom = (std_dev > 1e-8f) ? std_dev : 1.0f;
    for (size_t i = 0; i < n && i < CHIMERA_MAX_ROWS; ++i) {
        out[i] = (r1[i] - mean) / denom;
    }
    return 0;
}

static int execute_single_ins(
    const ChimeraInstruction* ins,
    float* const* regs,
    const float* const* feats,
    size_t n_feats,
    size_t n_rows
) {
    assert(ins != nullptr && regs != nullptr);
    assert(ins->out_reg < CHIMERA_MAX_REGS);
    float* out = regs[ins->out_reg];
    const uint8_t op = ins->op;
    if (op == CHIMERA_OP_NOP) {
        return 0;
    }
    if (op == CHIMERA_OP_LOAD_FEAT || op == CHIMERA_OP_LOAD_IMM) {
        const float* src = (op == CHIMERA_OP_LOAD_FEAT && ins->feat_idx < n_feats && feats != nullptr) ? feats[ins->feat_idx] : nullptr;
        return eval_load(out, src, ins->imm_val, op, n_rows);
    }
    if (op >= CHIMERA_OP_ADD && op <= CHIMERA_OP_SAFE_DIV) {
        return eval_arith(out, regs[ins->in_reg1], regs[ins->in_reg2], op, n_rows);
    }
    if (op == CHIMERA_OP_MAX || op == CHIMERA_OP_MIN) {
        return eval_extrema(out, regs[ins->in_reg1], regs[ins->in_reg2], op, n_rows);
    }
    if (op >= CHIMERA_OP_TANH && op <= CHIMERA_OP_LOG1P) {
        return eval_unary(out, regs[ins->in_reg1], op, n_rows);
    }
    if (op == CHIMERA_OP_ZSCORE) {
        return eval_zscore(out, regs[ins->in_reg1], n_rows);
    }
    return -1;
}

int chimera_execute(
    const ChimeraInstruction* instrs,
    size_t num_instrs,
    const float* const* feature_cols,
    size_t num_features,
    size_t num_rows,
    ChimeraArena* arena,
    float* output
) {
    assert(instrs != nullptr && arena != nullptr);
    assert(output != nullptr && arena->is_initialized == 1);
    assert(num_rows > 0 && num_rows <= arena->capacity_rows);
    assert(num_instrs > 0 && num_instrs <= CHIMERA_MAX_INSTRS);

    if (instrs == nullptr || arena == nullptr || output == nullptr || arena->is_initialized != 1) {
        return -1;
    }
    if (num_rows == 0 || num_rows > arena->capacity_rows || num_instrs == 0 || num_instrs > CHIMERA_MAX_INSTRS) {
        return -2;
    }

    float* regs[CHIMERA_MAX_REGS];
    for (size_t r = 0; r < CHIMERA_MAX_REGS; ++r) {
        regs[r] = arena->buffer + (r * arena->capacity_rows);
    }

    for (size_t k = 0; k < num_instrs && k < CHIMERA_MAX_INSTRS; ++k) {
        const int rc = execute_single_ins(&instrs[k], regs, feature_cols, num_features, num_rows);
        if (rc != 0) {
            return rc;
        }
    }

    const uint8_t final_reg = instrs[num_instrs - 1].out_reg;
    assert(final_reg < CHIMERA_MAX_REGS);
    std::memcpy(output, regs[final_reg], num_rows * sizeof(float));
    return 0;
}
