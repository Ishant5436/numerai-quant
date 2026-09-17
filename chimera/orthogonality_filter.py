import numpy as np
from scipy.stats import spearmanr
from dataclasses import dataclass

@dataclass
class HurdleResult:
    passed: bool
    sharpe: float
    mean_corr: float
    max_factor_corr: float
    positive_era_ratio: float
    failure_reason: str

class TriHurdleFilter:
    def __init__(
        self,
        min_sharpe: float = 1.05,
        max_factor_corr: float = 0.12,
        min_positive_era_ratio: float = 0.65
    ):
        assert min_sharpe > 0.0, "min_sharpe must be positive"
        assert max_factor_corr > 0.0 and max_factor_corr < 1.0, "max_factor_corr must be in (0, 1)"
        assert min_positive_era_ratio > 0.0 and min_positive_era_ratio <= 1.0, "min_positive_era_ratio must be in (0, 1]"
        
        self.min_sharpe = min_sharpe
        self.max_factor_corr = max_factor_corr
        self.min_positive_era_ratio = min_positive_era_ratio

    def evaluate(
        self,
        signal: np.ndarray,
        target: np.ndarray,
        eras: np.ndarray,
        feature_matrix: np.ndarray,
        max_features_to_check: int = 150
    ) -> HurdleResult:
        assert len(signal) == len(target) == len(eras) == len(feature_matrix), "Input lengths must match"
        
        # 1. Per-era performance
        unique_eras = np.unique(eras)
        era_corrs = []
        for era in unique_eras:
            mask = (eras == era)
            sig_era = signal[mask]
            tar_era = target[mask]
            
            if np.std(sig_era) < 1e-7 or np.std(tar_era) < 1e-7:
                era_corrs.append(0.0)
                continue
                
            corr, _ = spearmanr(sig_era, tar_era)
            if np.isnan(corr):
                corr = 0.0
            era_corrs.append(float(corr))

        era_corrs = np.array(era_corrs, dtype=np.float64)
        mean_corr = float(np.mean(era_corrs))
        std_corr = float(np.std(era_corrs))
        sharpe = float(mean_corr / (std_corr + 1e-8))
        positive_era_ratio = float(np.mean(era_corrs > 0.0))

        # 2. Orthogonality against base features
        n_feats = feature_matrix.shape[1]
        check_indices = np.random.choice(n_feats, min(n_feats, max_features_to_check), replace=False) if n_feats > max_features_to_check else np.arange(n_feats)
        
        max_factor_corr = 0.0
        for f_idx in check_indices:
            feat_col = feature_matrix[:, f_idx]
            if np.std(feat_col) < 1e-7:
                continue
            f_corr, _ = spearmanr(signal, feat_col)
            if not np.isnan(f_corr):
                f_abs = abs(float(f_corr))
                if f_abs > max_factor_corr:
                    max_factor_corr = f_abs

        failures = []
        if sharpe < self.min_sharpe:
            failures.append(f"Performance failure: Sharpe {sharpe:.3f} < min {self.min_sharpe:.3f}")
        if positive_era_ratio < self.min_positive_era_ratio:
            failures.append(f"Stability failure: positive era ratio {positive_era_ratio:.1%} < min {self.min_positive_era_ratio:.1%}")
        if max_factor_corr > self.max_factor_corr:
            failures.append(f"Orthogonality failure: max feature correlation {max_factor_corr:.3f} > max {self.max_factor_corr:.3f}")

        passed = (len(failures) == 0)
        return HurdleResult(
            passed=passed,
            sharpe=sharpe,
            mean_corr=mean_corr,
            max_factor_corr=max_factor_corr,
            positive_era_ratio=positive_era_ratio,
            failure_reason="; ".join(failures)
        )
