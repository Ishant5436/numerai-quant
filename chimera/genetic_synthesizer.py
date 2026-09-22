import random
import numpy as np
from typing import List, Optional
from chimera.ast_generator import ASTGenerator, Node
from chimera.evaluator import ChimeraEngine
from chimera.orthogonality_filter import TriHurdleFilter
from chimera.alpha_vault import AlphaVault, AlphaEntry

class GeneticSynthesizer:
    def __init__(
        self,
        num_features: int,
        pop_size: int = 50,
        tournament_size: int = 3,
        mutation_rate: float = 0.35,
        crossover_rate: float = 0.65,
        max_depth: int = 4,
        vault_path: str = "data/alpha_vault.json",
        filter_gate: Optional[TriHurdleFilter] = None
    ):
        assert num_features > 0, "num_features must be > 0"
        assert pop_size >= 4, "pop_size must be >= 4"
        self.num_features = num_features
        self.pop_size = pop_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_depth = max_depth
        self.vault_path = vault_path
        
        self.gen = ASTGenerator(num_features=num_features, max_depth=max_depth)
        self.filter_gate = filter_gate if filter_gate is not None else TriHurdleFilter()
        self.vault = AlphaVault.load(vault_path)

    def _evaluate_individual(
        self,
        tree: Node,
        engine: ChimeraEngine,
        features: np.ndarray,
        target: np.ndarray,
        eras: np.ndarray
    ) -> float:
        assert tree is not None, "tree must not be None"
        assert len(features) == len(target), "features and target lengths must match"
        try:
            instructions = tree.compile_to_bytecode()
            output = engine.execute(instructions, features)
            
            if not np.isfinite(output).all() or np.std(output) < 1e-6:
                return -1.0
            
            corr = np.corrcoef(output, target)[0, 1]
            if np.isnan(corr) or corr <= 0.0:
                return -1.0
            
            feat_indices = set(tree.get_feature_indices())
            max_f_corr = 0.0
            for fi in feat_indices:
                if fi < features.shape[1]:
                    f_col = features[:, fi]
                    if np.std(f_col) > 1e-6:
                        fc = abs(float(np.corrcoef(output, f_col)[0, 1]))
                        if not np.isnan(fc) and fc > max_f_corr:
                            max_f_corr = fc
            
            penalty = 1.0 + 4.0 * max(0.0, max_f_corr - 0.15)
            return float(corr / penalty)
        except Exception:
            return -1.0

    def _evaluate_population_and_harvest(
        self,
        population: List[Node],
        engine: ChimeraEngine,
        features: np.ndarray,
        target: np.ndarray,
        eras: np.ndarray,
        gen_idx: int
    ) -> List[float]:
        assert len(population) == self.pop_size, "Population size mismatch"
        assert len(features) == len(target) == len(eras), "Data lengths must match"
        
        fitness_scores = [self._evaluate_individual(ind, engine, features, target, eras) for ind in population]
        top_indices = np.argsort(fitness_scores)[::-1][:15]
        
        for idx in top_indices:
            cand = population[idx]
            cand_instrs = cand.compile_to_bytecode()
            sig = engine.execute(cand_instrs, features)
            
            hurdle_res = self.filter_gate.evaluate(sig, target, eras, features)
            if hurdle_res.passed:
                formula_str = cand.to_formula()
                if any(e.formula == formula_str for e in self.vault.entries):
                    continue
                entry_idx = len(self.vault.entries) + 1
                alpha_entry = AlphaEntry(
                    name=f"chimera_alpha_{entry_idx:02d}",
                    formula=formula_str,
                    instructions=[(i.op, i.out_reg, i.in_reg1, i.in_reg2, i.feat_idx, i.imm_val) for i in cand_instrs],
                    sharpe=hurdle_res.sharpe,
                    mean_corr=hurdle_res.mean_corr,
                    max_factor_corr=hurdle_res.max_factor_corr,
                    positive_era_ratio=hurdle_res.positive_era_ratio
                )
                self.vault.add_entry(alpha_entry)
                self.vault.save()
                print(f"[*] Discovery [Gen {gen_idx}]: {alpha_entry.name} -> {alpha_entry.formula} (Sharpe={hurdle_res.sharpe:.3f}, Corr={hurdle_res.mean_corr:.4f})")
                
        return fitness_scores

    def _produce_next_generation(self, population: List[Node], fitness_scores: List[float]) -> List[Node]:
        assert len(population) == len(fitness_scores) == self.pop_size, "Mismatch in population/scores"
        assert self.tournament_size > 0, "Invalid tournament size"
        
        top_indices = np.argsort(fitness_scores)[::-1]
        next_gen: List[Node] = [population[i].clone() for i in top_indices[:2]]
        
        while len(next_gen) < self.pop_size:
            p1_candidates = random.sample(range(self.pop_size), self.tournament_size)
            p1_idx = max(p1_candidates, key=lambda i: fitness_scores[i])
            p1 = population[p1_idx]

            if random.random() < self.crossover_rate:
                p2_candidates = random.sample(range(self.pop_size), self.tournament_size)
                p2_idx = max(p2_candidates, key=lambda i: fitness_scores[i])
                p2 = population[p2_idx]
                c1, c2 = self.gen.crossover(p1, p2)
            else:
                c1, c2 = p1.clone(), p1.clone()

            if random.random() < self.mutation_rate:
                c1 = self.gen.mutate(c1)
            if random.random() < self.mutation_rate:
                c2 = self.gen.mutate(c2)

            next_gen.append(c1)
            if len(next_gen) < self.pop_size:
                next_gen.append(c2)
                
        return next_gen

    def evolve_on_dataset(
        self,
        features: np.ndarray,
        target: np.ndarray,
        eras: np.ndarray,
        generations: int = 5
    ) -> AlphaVault:
        assert len(features) == len(target) == len(eras), "Data lengths must match"
        assert generations > 0, "generations must be > 0"
        features = np.ascontiguousarray(features, dtype=np.float32)
        target = np.ascontiguousarray(target, dtype=np.float32)
        n_rows = len(features)
        engine = ChimeraEngine(capacity_rows=max(1000, n_rows))
        
        population: List[Node] = [self.gen.random_tree() for _ in range(self.pop_size)]

        try:
            for gen_idx in range(generations):
                fitness_scores = self._evaluate_population_and_harvest(
                    population, engine, features, target, eras, gen_idx
                )
                population = self._produce_next_generation(population, fitness_scores)
        finally:
            engine.close()

        self.vault.save()
        return self.vault

