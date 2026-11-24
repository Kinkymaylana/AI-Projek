# Paper Draft Materials — Member B (Method: Modeling & Simulated Annealing, and Validation Results)

Prepared for IEEE-format mini-paper integration (English). All standard deviations reported in this document use **sample standard deviation (`ddof=1`)** for consistency with Member C. All model selection, hyperparameter tuning, and feature selection were performed strictly on `train` (actors 1–16) and `validation` (actors 17–20); the held-out `test` split (actors 21–24) remained untouched.

---

## III-E. Baseline Modeling and Amplitude Normalization Ablation

To evaluate the 90 extracted acoustic features without data leakage, we wrapped feature standardization (`StandardScaler`) and classification inside a single scikit-learn `Pipeline`, ensuring scaling statistics were fitted exclusively on training folds. Because the `neutral` class contains half as many utterances as the other seven emotions (96 vs. 192), all classifiers used `class_weight='balanced'` and were optimized using **Macro-F1**.

We evaluated two classifier families on the full 90-feature set: Support Vector Machine (`SVC`, tuning RBF and linear kernels over $C \in \{0.01, \dots, 20\}$ and $\gamma \in \{\text{scale}, \text{auto}, 0.01, 0.005, 0.001\}$) and Random Forest (`RandomForestClassifier`, tuning `n_estimators`, `max_depth`, `min_samples_split`, and `max_features`). Hyperparameters were selected via 4-fold speaker-independent `GroupKFold` on the 16 training actors, and the resulting best models were evaluated once on the 4 unseen validation actors (actors 17–20) to compare unnormalized (`none`) versus peak-normalized (`peak`) audio features.

**Table I — 90-Feature Baseline Comparison on Validation (Actors 17–20):**

| Normalization | Model | Best Hyperparameters (Train 4-Fold GroupKFold) | Train 4-Fold CV Macro-F1 | **Val Macro-F1** | Val UAR | Val Accuracy |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `none` | SVM | RBF, $C=5.0, \gamma=0.001$ | 0.5034 | 0.4846 | 0.5039 | 0.4833 |
| `none` | RF | 100 trees, `max_depth=10`, `max_features=0.3` | 0.4517 | 0.4316 | 0.4414 | 0.4625 |
| **`peak`** | **SVM** | **RBF, $C=1.0, \gamma=0.005$** | **0.4901** | **0.5165** | **0.5352** | **0.5167** |
| **`peak`** | **RF** | 400 trees, `max_depth=None`, `max_features=0.3`, `min_samples_split=5` | 0.4421 | 0.4644 | 0.4688 | 0.4875 |

**Key finding:** Peak normalization consistently improved unseen-speaker validation Macro-F1 by **+0.0319** for SVM (`0.4846 → 0.5165`) and **+0.0328** for RF (`0.4316 → 0.4644`). Although `none` achieved slightly higher CV Macro-F1 within the training actors, inter-actor loudness variance (18.0 dB median range) caused unnormalized energy features to overfit speaker volume rather than emotion. Therefore, **`peak`** normalization was locked for all subsequent experiments (Fig. `progres_anggota_b/figures/sa_en/baseline_none_vs_peak_en.png`).

---

## III-F. Simulated Annealing Feature Selection and $\lambda$-Sweep

We formulated feature selection as binary subset optimization over $s \in \{0, 1\}^{90}$ ($|s| \ge 5$). To prevent overfitting to the 4 validation actors, the Simulated Annealing (SA) objective was evaluated internally via 4-fold `GroupKFold` on the 16 training actors:
$$J(s) = \text{Macro-F1}_{\text{4-fold CV}}(s) - \lambda \cdot \frac{|s|}{90}$$
where $\lambda$ penalizes subset cardinality. Inside the SA loop, we used `SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')` so that the RBF kernel scale $\gamma = 1/(k \cdot \text{Var})$ automatically adapted to the active subset size $k = |s|$. Standard scaling was precomputed independently per CV fold (strictly fitted on each fold's training split), yielding mathematically identical results to per-subset `Pipeline` fitting with zero leakage.

SA ran for 500 iterations with geometric cooling ($T_0 = 0.015 \to T_{\min} = 2\times 10^{-4}$), adaptive neighborhood moves (1–3 bit flips at high temperature $T > 0.3 T_0$; single bit flip or active/inactive swap at low temperature), Metropolis acceptance $\exp(-\Delta E / T)$, and stagnation reheating (after 80 non-improving steps, up to 2 reheats). After SA converged, the SVM classifier was re-tuned on the selected subset $S^*$ via 4-fold `GroupKFold` on train.

### Penalty Weight ($\lambda$) Sensitivity Sweep
We swept $\lambda \in \{0.0, 0.005, 0.015, 0.03, 0.05, 0.08\}$ on the main split `P0` (seed 42):

**Table II — Parsimony Penalty ($\lambda$) Sweep on Main Split `P0`:**

| $\lambda$ | Selected $k$ | Reduction (%) | Train 4-Fold CV F1 | **Val Macro-F1** | Val UAR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 48 | 46.7% | 0.5243 | 0.4940 | 0.5039 |
| 0.005 | 52 | 42.2% | 0.5224 | **0.4995** | **0.5078** |
| **0.015** | **52** | **42.2%** | **0.5224** | **0.4995** | **0.5078** |
| 0.030 | 48 | 46.7% | 0.5160 | 0.4860 | 0.5039 |
| 0.050 | 51 | 43.3% | 0.5120 | 0.4892 | 0.4922 |
| 0.080 | 36 | 60.0% | 0.5082 | 0.4493 | 0.4531 |

We selected **$\lambda = 0.015$**, which attained the highest validation Macro-F1 (`0.4995`) while eliminating 42.2% of features ($90 \to 52$). Stronger penalties ($\lambda \ge 0.03$) degraded both CV and validation scores, with a sharp drop at $\lambda = 0.08$ (`0.4493`) due to aggressive removal of discriminative descriptors (Fig. `progres_anggota_b/figures/sa_en/lambda_sweep_tradeoff_en.png`).

---

## III-G. Comparators, Final Model Retraining, and Validation Summary

To benchmark SA fairly against simpler selection and modeling strategies on the exact same `P0` train/validation split (`peak` normalization), we evaluated:
1. **Mutual Information (`SVM_MI52`):** top-$k$ ($k=52$) univariate filter features ranked on train only (`data/analysis/mi_scores_peak.csv`), with SVM hyperparameters tuned via 4-fold `GroupKFold` on train (`linear, C=0.05`).
2. **Hill Climbing (`SVM_HC48`):** greedy local search with identical budget and objective (`k=48`, tuned RBF `C=1.0, gamma='scale'`).
3. **Random Subsets (`Random52`):** 30 random 52-feature subsets to establish a chance-selection baseline.
4. **Soft-Voting Ensemble (`SVM + RF`):** combining SVM and RF probability predictions on both all 90 features and the 52 SA features.
5. **Random Forest (`RF_SA52`):** intentionally reusing the best `RF_all90` hyperparameters without re-tuning for a controlled feature-subset comparison.

**Table III — Validation Performance Summary on Main Split `P0` (Actors 17–20):**

| Method | $k$ | Train 4-Fold CV F1 | **Val Macro-F1** | Val UAR |
| --- | ---: | ---: | ---: | ---: |
| `SVM_all90` (Baseline) | 90 | 0.4901 | 0.5165 | 0.5352 |
| `SVM_SA52` (Proposed SA, seed 42) | 52 | **0.5224** | 0.4995 | 0.5078 |
| `SVM_SA` (P0 3-seed mean ± SD, `ddof=1`) | 52.67 ± 1.15 | 0.5100 ± 0.0114 | 0.4972 ± 0.0130 | — |
| `SVM_MI52` (Mutual Information top-52) | 52 | 0.4718 | 0.4257 | 0.4375 |
| `SVM_HC48` (Hill Climbing, seed 42) | 48 | 0.5118 | 0.5025 | 0.5195 |
| `SVM_Random52` (30 random subsets, mean ± SD) | 52 | 0.4276 ± 0.0200 | 0.4736 ± 0.0273 | — |
| `RF_all90` | 90 | 0.4421 | 0.4644 | 0.4688 |
| `RF_SA52` | 52 | — | 0.4522 | 0.4648 |
| `Ensemble_SVM_RF_all90` | 90 | 0.4640 | **0.5239** | 0.5195 |
| `Ensemble_SVM_RF_SA52` | 52 | 0.4886 | 0.5055 | 0.5078 |

**Validation takeaways:**
- On equal budget ($k=52$), **SA (`0.4995`) substantially outperformed univariate MI (`0.4257`, +0.0738)** and random subsets (`0.4736`, +0.0259), demonstrating that multivariate wrapper optimization captures feature interactions missed by univariate filtering (which tends to over-select redundant clusters).
- Compared to the 90-feature SVM baseline (`0.5165`), SA improved internal train CV F1 (`0.4901 → 0.5224`) while removing 42.2% of features, with a minor `-0.0170` drop on the 4 validation actors—reflecting mild wrapper overfitting across limited speaker groups (16 training actors).
- Across the 15 stability runs (`P0`–`P4` × 3 seeds), SA selected $k = 46.33 \pm 5.79$ features (`ddof=1`), with train CV Macro-F1 $0.5133 \pm 0.0176$ and validation Macro-F1 $0.4734 \pm 0.0296$. The most consistently selected acoustic descriptors were `voiced_ratio` (15/15 runs), `mfcc03_std`, `mfcc04_std`, and `dmfcc04_std` (14/15 runs each).

### Final Model Retraining (`train + validation`, Actors 1–20)
Before final single-shot test evaluation by Member C, all final models were refitted on the combined 1,200 `train + validation` utterances (actors 1–20) with `probability=True` for calibrated Gradio demo outputs. Diagnostic 5-fold speaker-independent `GroupKFold` Macro-F1 (`mean ± SD`, `ddof=1`) on `train + validation` was: `SVM_all90` `0.4868 ± 0.0501`, `SVM_SA52` **`0.5034 ± 0.0644`**, `SVM_MI52` `0.4977 ± 0.0501`, `RF_all90` `0.4292 ± 0.0400`, and `RF_SA52` `0.4394 ± 0.0498`.

### Limitations to Disclose in Paper
Due to computational budget constraints, the main-split SA experiment used 3 random seeds (vs. 5 originally planned), while Hill Climbing and the $\lambda$-sweep were run with 1 seed (`42`) each; all seed counts are reported explicitly for full reproducibility.

---

## English Figures Ready for IEEE Paper (`progres_anggota_b/figures/sa_en/`)
1. `baseline_none_vs_peak_en.png` — Normalization ablation (`none` vs. `peak`, SVM & RF).
2. `sa_main_convergence_en.png` — SA convergence trace (CV F1, objective $J(s)$, and subset size $k$).
3. `sa_main_feature_groups_en.png` — Selected feature distribution across the 8 acoustic groups.
4. `lambda_sweep_tradeoff_en.png` — Trade-off curve across $\lambda \in [0, 0.08]$.
5. `sa_stability_frequency_en.png` — Top 25 most frequently selected features across 15 stability runs.
