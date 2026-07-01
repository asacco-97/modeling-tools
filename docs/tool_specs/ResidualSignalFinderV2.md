Residual Signal Finder V2: Stability, Univariate Lift, and Diagnostic Upgrade Spec
Purpose
Update the residual signal finder class so it can more reliably identify features that explain remaining model residuals, and create a new version called ResidualSignalFinderV2
The package should take:
`actuals`: true target values
`base_predictions`: preferably out-of-fold predictions from the original/base model
`features`: candidate variables to evaluate for residual signal
optional metadata such as split columns, segment columns, weights, feature types, and original model feature list
The package should answer four questions:
Is there residual signal left?
Which features explain residuals out-of-sample?
Is the feature-residual relationship stable across bootstraps/splits?
What should the modeler do with the feature?
include as a new feature
respecify an existing feature
model nonlinearities
model interactions
model missingness
model variance/heteroskedasticity
ignore as unstable/noisy
The key upgrade is to move away from relying primarily on gain-based importance from a multivariate tree model. Instead:
Use a multivariate residual model only as an optional screening stage.
Use univariate residual lift models as the primary feature scoring method.
Use permutation importance instead of gain-based importance.
Use objective stability summaries:
out-of-fold residual R²
positive uplift rate
median rank
top-5 rate
effect-curve stability
median Spearman
percent positive Spearman
null/shadow feature baseline
---
Core Concept
Define residuals as:
```python
residual = actuals - base_predictions
```
The package should assume that `base_predictions` are already out-of-sample or out-of-fold predictions whenever possible. If the predictions are in-sample, the package should warn the user that residual diagnostics may be biased by base-model overfit.
---
Recommended Workflow
Stage 0: Input Validation
Validate:
`actuals` and `base_predictions` have the same length.
`features` has the same row count.
No duplicate feature names.
Feature names are valid and stable.
Optional sample weights align with rows.
Optional split column aligns with rows.
Optional segment columns align with rows.
Residuals are finite after filtering.
Create:
```python
residuals = actuals - base_predictions
```
Also compute global residual summaries:
```python
mean_residual
median_residual
std_residual
mae_residual
rmse_residual
residual_skew
residual_kurtosis
```
---
Stage 1: Optional Multivariate Screening Model
Purpose
When the number of candidate features is large, fitting a univariate model for every feature across many bootstraps may be unnecessary. Use a shallow multivariate model to identify a candidate set of top features.
This stage should be optional.
Config
```python
screening_enabled: bool = True
screening_model_type: Literal["xgboost", "random_forest"] = "xgboost"
screening_top_k: int = 50
screening_cv_folds: int = 5
screening_n_repeats: int = 3
screening_metric: Literal["permutation_importance", "oof_r2"] = "permutation_importance"
```
Model Defaults
For XGBoost:
```python
max_depth = 1 or 2
n_estimators = 100 to 300
learning_rate = 0.03 to 0.10
subsample = 0.8
colsample_bytree = 0.8
min_child_weight = conservative default
reg_lambda = 1.0+
early_stopping_rounds = optional
```
For random forest:
```python
max_depth = 2 to 4
min_samples_leaf = conservative default
n_estimators = 300+
max_features = "sqrt" or lower
bootstrap = True
```
Importance Method
Use permutation importance, not gain-based feature importance.
For each validation fold:
Fit model on training split.
Predict residuals on validation split.
Compute validation metric.
For each feature, permute the feature in the validation set.
Recompute validation metric.
Importance is the drop in performance.
For residual regression, default metric:
```python
oof_r2 = 1 - SSE_model / SSE_null
```
Where:
```python
SSE_model = sum((residual_val - pred_residual_val) ** 2)
SSE_null = sum((residual_val - mean_residual_train) ** 2)
```
Permutation importance:
```python
perm_importance_j = baseline_oof_r2 - permuted_oof_r2_j
```
Screening Output
For each feature:
```python
screening_mean_perm_importance
screening_median_perm_importance
screening_importance_std
screening_median_rank
screening_top_5_rate
screening_top_10_rate
```
Select top `screening_top_k` features by:
```python
mean_rank_or_median_rank
```
Recommended default:
```python
primary_screening_sort = [
    "screening_median_rank",
    "screening_top_5_rate",
    "screening_mean_perm_importance"
]
```
Important Caveat
The multivariate screening model is only a candidate generator.
The final feature evaluation should be based on univariate residual lift and stability, not the multivariate screening importance alone.
---
Stage 2: Univariate Residual Lift Models
Purpose
For each candidate feature, fit a separate model:
```python
residual ~ feature_j
```
This provides a clean estimate of the feature's standalone residual signal.
This should be the primary scoring stage.
Candidate Feature Set
If screening is enabled:
```python
candidate_features = top screening_top_k features
```
If screening is disabled:
```python
candidate_features = all features
```
Supported Model Types
```python
univariate_model_type: Literal["xgboost", "random_forest", "tree", "spline", "isotonic"] = "xgboost"
```
Initial implementation should support:
XGBoost
Random forest
Future options:
decision tree stump/shallow tree
spline/GAM smoother
isotonic regression for monotone effects
Recommended Defaults
For univariate XGBoost:
```python
max_depth = 1 or 2
n_estimators = 100 to 300
learning_rate = 0.03 to 0.10
subsample = 0.8
min_child_weight = conservative
reg_lambda = 1.0+
```
For univariate random forest:
```python
max_depth = 2 to 4
n_estimators = 300+
min_samples_leaf = max(20, 0.02 * n_train)
bootstrap = True
```
Use depth 1 when you want main-effect monotone or threshold detection.
Use depth 2 when you want a univariate model with slightly more flexible nonlinear shape.
---
Stage 3: Bootstrap / Repeated Split Procedure
Purpose
Estimate both signal strength and stability.
Config
```python
n_bootstraps: int = 100
test_size: float = 0.2
split_strategy: Literal["bootstrap", "repeated_kfold", "user_split", "group_kfold"] = "bootstrap"
random_state: int = 42
stratify_column: Optional[str] = None
group_column: Optional[str] = None
weight_column: Optional[str] = None
```
Procedure
For each bootstrap/split `b`:
Create train/validation split.
For each candidate feature `j`:
fit univariate model on training residuals
predict residuals on validation rows
compute validation residual R²
compute Spearman correlation between feature and residual on validation rows
compute effect curve on validation rows
Rank features within bootstrap by validation residual R² or permutation importance.
Store all per-bootstrap results.
---
Primary Univariate Metric: Out-of-Fold Residual R²
For feature `j` and bootstrap `b`:
```python
sse_model = sum((residual_val - pred_residual_val_j) ** 2)
sse_null = sum((residual_val - mean(residual_train)) ** 2)

oof_r2_j_b = 1 - sse_model / sse_null
```
If sample weights are supplied:
```python
sse_model = sum(weight_val * (residual_val - pred_residual_val_j) ** 2)
sse_null = sum(weight_val * (residual_val - weighted_mean_residual_train) ** 2)
```
Aggregate Metrics
For each feature:
```python
mean_oof_r2
median_oof_r2
p10_oof_r2
p25_oof_r2
p75_oof_r2
p90_oof_r2
std_oof_r2
```
Positive Uplift Rate
Define a small threshold:
```python
r2_epsilon = 0.001
```
Then:
```python
positive_uplift_rate = mean(oof_r2_j_b > r2_epsilon)
```
This answers:
> In what percentage of resamples did this feature produce practically positive validation lift?
Recommended thresholds:
```python
strong_positive_uplift_rate >= 0.75
watchlist_positive_uplift_rate >= 0.55
weak_positive_uplift_rate < 0.55
```
---
Rank Metrics
Within each bootstrap, rank features by validation residual R² or permutation importance.
Default:
```python
rank_metric = "oof_r2"
```
For each feature:
```python
mean_rank
median_rank
rank_std
rank_iqr
top_5_rate
top_10_rate
```
The user specifically wants:
```python
median_rank
top_5_rate
```
These should be included in the primary summary table.
Top-5 Rate
```python
top_5_rate = mean(rank_j_b <= 5)
```
Median Rank
```python
median_rank = median(rank_j_b)
```
Rank is useful because absolute R² can be tiny when residuals are noisy. A feature can still be useful if it consistently ranks near the top.
---
Direction Stability
Compute Spearman correlation between the raw feature and residual on validation rows.
For each feature and bootstrap:
```python
spearman_j_b = spearmanr(feature_j_val, residual_val)
```
Handle:
constant features
all-missing features
too few unique values
categorical variables where Spearman may be inappropriate
Aggregate Metrics
```python
median_spearman
mean_spearman
percent_positive_spearman
percent_negative_spearman
spearman_iqr
```
The primary direction metrics should be:
```python
median_spearman
percent_positive_spearman
```
Where:
```python
percent_positive_spearman = mean(spearman_j_b > 0)
```
Interpretation:
`median_spearman > 0` and `percent_positive_spearman >= 0.75`: stable positive monotone tendency
`median_spearman < 0` and `percent_positive_spearman <= 0.25`: stable negative monotone tendency
`percent_positive_spearman around 0.5`: unstable or non-monotone
Important:
Spearman only measures monotone association. A stable U-shape or threshold effect may have low Spearman but high effect-curve stability.
Therefore, do not use Spearman as the main stability score.
---
Effect Curve Stability
Purpose
Effect-curve stability is the key new stability metric.
It should answer:
> Does the residual-vs-feature relationship have a similar shape across bootstraps?
This captures:
monotone relationships
nonlinear relationships
U-shapes
threshold effects
tail effects
sparse pockets of underprediction/overprediction
Continuous Features
For each continuous feature:
Create fixed quantile bins using the full dataset or training data.
Recommended default number of bins:
```python
n_bins = 10
```
Use these same bin cut points across all bootstraps.
For each bootstrap validation set, compute the mean residual per bin.
Center the curve by subtracting the validation-set mean residual.
For feature `j`, bootstrap `b`, bin `q`:
```python
curve_j_b_q = mean(residual_val in bin q) - mean(residual_val)
```
If using sample weights:
```python
curve_j_b_q = weighted_mean(residual_val in bin q) - weighted_mean(residual_val)
```
Optionally standardize by residual standard deviation:
```python
curve_j_b_q_standardized = curve_j_b_q / std(residual_val)
```
Categorical Features
For categorical features:
Use top `max_categories` categories by frequency.
Group rare categories into `"__OTHER__"`.
Keep missing as `"__MISSING__"`.
Compute mean centered residual by category.
Use the same category ordering across bootstraps.
Recommended defaults:
```python
max_categories = 20
min_category_count = 30
rare_category_label = "__OTHER__"
missing_category_label = "__MISSING__"
```
Shape Stability
Each bootstrap produces a vector:
```python
curve_j_b = [curve_j_b_1, curve_j_b_2, ..., curve_j_b_Q]
```
Compute pairwise correlations between bootstrap curves.
```python
shape_corrs = [
    corr(curve_j_b1, curve_j_b2)
    for all pairs b1, b2
]
```
Then:
```python
effect_curve_stability = median(shape_corrs)
```
Clip to `[0, 1]`:
```python
effect_curve_stability = max(0, median(shape_corrs))
```
If too few valid bins exist, set:
```python
effect_curve_stability = np.nan
effect_curve_valid = False
```
Curve Signal-to-Noise
Also compute:
```python
mean_curve_q = mean(curve_j_b_q across bootstraps)
std_curve_q = std(curve_j_b_q across bootstraps)

curve_snr = std(mean_curve_q) / mean(std_curve_q)
```
Interpretation:
high curve SNR means the average pattern is large relative to bootstrap noise
low curve SNR means the curve is mostly noise
Curve Effect Size
Compute:
```python
curve_effect_size = max(mean_curve_q) - min(mean_curve_q)
```
Optional standardized version:
```python
curve_effect_size_std = curve_effect_size / std(residual)
```
This helps distinguish stable but tiny curves from stable and meaningful curves.
---
Null / Shadow Feature Baseline
Purpose
Prevent false positives in small, sparse, noisy data.
The package should create null features and evaluate them through the same pipeline as real features.
Supported options:
```python
null_strategy: Literal["permuted_features", "random_noise", "shuffled_residuals"] = "permuted_features"
n_null_features: int = min(20, n_candidate_features)
```
Recommended default:
```python
null_strategy = "permuted_features"
```
Permuted Feature Nulls
For a random subset of real features:
```python
shadow_feature_j = permutation(feature_j)
```
These preserve marginal distributions but break the relationship with residuals.
Null Metrics
For each null/shadow feature, compute:
```python
mean_oof_r2
positive_uplift_rate
effect_curve_stability
curve_effect_size
median_rank
top_5_rate
```
Null Beat Rate
For each real feature and bootstrap:
```python
real_score_j_b > percentile(null_scores_b, 95)
```
Then:
```python
null_beat_rate = mean(real_score_j_b > null_95th_percentile_b)
```
Recommended scoring variable:
```python
score_for_null_comparison = oof_r2_j_b
```
Optional secondary null comparison:
```python
curve_effect_size_j_b
```
Interpretation
```python
null_beat_rate >= 0.80: strong evidence feature beats noise
null_beat_rate >= 0.60: moderate evidence
null_beat_rate < 0.50: likely noise or unstable
```
---
Composite Stability Score
Expose all components separately, but also provide a composite score.
Required Inputs
For each feature:
```python
positive_uplift_rate
effect_curve_stability
top_5_rate
rank_stability
spearman_direction_score
null_beat_rate
normalized_signal_strength
```
Rank Stability
Since the user only requires median rank and top-5 rate, rank stability can be optional.
If implemented:
```python
rank_stability = 1 - (rank_iqr / max_possible_rank_range)
rank_stability = clip(rank_stability, 0, 1)
```
Direction Score
Use percent positive Spearman:
```python
direction_score = abs(percent_positive_spearman - 0.5) * 2
```
This ranges from 0 to 1.
Examples:
```python
percent_positive_spearman = 0.50 -> direction_score = 0.00
percent_positive_spearman = 0.75 -> direction_score = 0.50
percent_positive_spearman = 0.90 -> direction_score = 0.80
percent_positive_spearman = 1.00 -> direction_score = 1.00
```
However, direction score should be treated as secondary because nonlinear relationships may not be monotone.
Recommended Composite
```python
stability_score = (
    0.35 * positive_uplift_rate
    + 0.30 * effect_curve_stability
    + 0.20 * top_5_rate
    + 0.15 * direction_score
)
```
Then incorporate the null baseline:
```python
credible_stability_score = stability_score * null_beat_rate
```
Signal Strength
Normalize `mean_oof_r2` across features:
```python
normalized_signal_strength = percentile_rank(mean_oof_r2)
```
or use:
```python
normalized_signal_strength = min(mean_oof_r2 / strong_r2_threshold, 1.0)
```
Default:
```python
strong_r2_threshold = 0.02
```
This should be configurable.
Final Actionability Score
```python
actionability_score = credible_stability_score * normalized_signal_strength
```
The package should report both:
```python
credible_stability_score
actionability_score
```
Do not hide component metrics.
---
Recommendation Labels
The package should assign an interpretable recommendation.
Strong Candidate
```python
if (
    positive_uplift_rate >= 0.75
    and effect_curve_stability >= 0.65
    and null_beat_rate >= 0.80
):
    label = "strong_residual_signal"
```
Respecification Candidate
If feature was in original model:
```python
if label == "strong_residual_signal" and feature in original_model_features:
    recommendation = "respecify_existing_feature"
```
Examples:
add spline
add piecewise term
transform feature
cap/winsorize differently
add interaction
separate missingness treatment
Include Candidate
If feature was not in original model:
```python
if label == "strong_residual_signal" and feature not in original_model_features:
    recommendation = "include_candidate"
```
Interaction Candidate
If feature has weak univariate signal but appears important in depth-2 multivariate residual model:
```python
recommendation = "interaction_candidate"
```
This should be optional and based on a separate interaction diagnostic.
Variance Candidate
If feature predicts absolute residual or squared residual more than signed residual:
```python
recommendation = "variance_candidate"
```
Watchlist
```python
if (
    positive_uplift_rate >= 0.55
    and effect_curve_stability >= 0.50
    and null_beat_rate >= 0.60
):
    recommendation = "watchlist"
```
Likely Noise
```python
if null_beat_rate < 0.50 or positive_uplift_rate < 0.50:
    recommendation = "likely_noise"
```
Unstable Signal
```python
if mean_oof_r2 is high but effect_curve_stability is low:
    recommendation = "unstable_signal"
```
---
Primary Output Table
The final feature-level summary should include:
```python
feature
feature_type
n_obs
n_missing
missing_rate

# residual lift
mean_oof_r2
median_oof_r2
p10_oof_r2
p90_oof_r2
positive_uplift_rate

# rank
median_rank
top_5_rate
top_10_rate

# direction
median_spearman
percent_positive_spearman

# effect curve
effect_curve_stability
curve_snr
curve_effect_size
curve_effect_size_std

# null comparison
null_beat_rate

# scores
stability_score
credible_stability_score
actionability_score

# recommendation
relationship_shape
recommendation
```
Sort default:
```python
sort_by = [
    "actionability_score",
    "credible_stability_score",
    "mean_oof_r2",
    "top_5_rate"
]
```
Descending except for rank metrics.
---
Feature-Level Diagnostics
Implement a method such as:
```python
plot_feature_diagnostics(feature_name)
```
or:
```python
diagnostics.plot_feature(feature_name)
```
It should create a multi-panel figure or return separate plot objects.
Required Plots
1. Actual vs Base Prediction vs Corrected Prediction by Feature Bin
For each feature bin/category:
actual mean
base prediction mean
base prediction + predicted residual mean
This shows whether the residual correction improves calibration.
For continuous features:
```python
x-axis = feature quantile bin
y-axis = mean target/prediction
lines = actual, base prediction, corrected prediction
```
For categorical features:
```python
x-axis = category
y-axis = mean target/prediction
```
Include count/exposure per bin.
2. Mean Residual by Feature Bin
```python
x-axis = feature bin/category
y-axis = mean residual
ribbon/error bars = bootstrap interval
```
This is the main misspecification plot.
3. Residual Model Effect Curve
```python
x-axis = feature bin/value/category
y-axis = predicted residual from univariate residual model
ribbon/error bars = bootstrap interval
```
This shows what the residual model learned.
4. Bootstrap Spaghetti Effect Curve
```python
thin lines = individual bootstrap residual curves
thick line = average residual curve
ribbon = 80/90/95% interval
```
This is the main visual diagnostic for effect-curve stability.
5. Residual Distribution by Feature Bin
Use boxplot/violin/strip plot depending on sample size.
```python
x-axis = feature bin/category
y-axis = residual
```
This helps distinguish mean shift from variance shift or outlier-driven signal.
6. Absolute Residual by Feature Bin
```python
x-axis = feature bin/category
y-axis = mean absolute residual
```
This identifies heteroskedasticity or uncertainty-model candidates.
7. Null Comparison Plot
Compare real feature score distribution against null/shadow feature score distribution.
```python
real bootstrap oof_r2 distribution
null bootstrap oof_r2 distribution
```
Optional:
```python
real curve_effect_size distribution
null curve_effect_size distribution
```
8. Missingness Diagnostic
If feature has missing values:
```python
mean residual when missing
mean residual when observed
mean absolute residual when missing
mean absolute residual when observed
count missing
count observed
```
---
Optional Diagnostics
Segment-Level Stability
Allow user to pass:
```python
segment_cols = ["year", "sector", "state", "vintage", "product"]
```
For each top feature, show residual curve by segment.
Useful plot:
```python
feature bin on x-axis
mean residual on y-axis
separate line for each segment
```
This answers:
> Is the residual relationship stable across meaningful business/time segments?
Interaction Heatmap
For top features, optionally evaluate pairwise residual surfaces.
```python
x-axis = feature A bins
y-axis = feature B bins
cell value = mean residual
cell annotation = count
```
This is useful when a shallow depth-2 multivariate model finds signal that univariate models do not fully explain.
---
API Design
Main Estimator
Create a class:
```python
class ResidualSignalFinder:
    def __init__(
        self,
        screening_enabled=True,
        screening_model_type="xgboost",
        screening_top_k=50,
        univariate_model_type="xgboost",
        n_bootstraps=100,
        test_size=0.2,
        r2_epsilon=0.001,
        n_bins=10,
        null_strategy="permuted_features",
        n_null_features=20,
        random_state=42,
        max_categories=20,
        min_category_count=30,
        use_sample_weight=False,
    ):
        ...
```
Fit Method
```python
finder.fit(
    X=features,
    y=actuals,
    base_pred=base_predictions,
    original_model_features=None,
    sample_weight=None,
    split_col=None,
    group_col=None,
    segment_cols=None,
)
```
Outputs
```python
finder.summary_
finder.bootstrap_results_
finder.effect_curves_
finder.null_results_
finder.screening_results_
```
Methods
```python
finder.get_summary()
finder.get_feature_summary(feature_name)
finder.plot_feature_diagnostics(feature_name)
finder.plot_top_features(n=10)
finder.plot_effect_curve(feature_name)
finder.plot_null_comparison(feature_name)
finder.plot_rank_stability(n=20)
finder.plot_residual_signal_map()
```
---
Internal Data Structures
`summary_`
One row per evaluated feature.
`bootstrap_results_`
One row per feature per bootstrap.
Required columns:
```python
bootstrap_id
feature
oof_r2
rank
spearman
is_top_5
is_top_10
null_95_score
beats_null_95
```
`effect_curves_`
One row per feature per bootstrap per bin/category.
Required columns:
```python
feature
bootstrap_id
bin_id
bin_label
bin_left
bin_right
n_obs
mean_residual
centered_mean_residual
predicted_residual
```
`null_results_`
One row per null feature per bootstrap.
Required columns:
```python
bootstrap_id
null_feature
source_feature
oof_r2
rank
curve_effect_size
effect_curve_stability
```
---
Handling Sparse Data
The package should be conservative with sparse features.
Minimum Validation Count
For each feature/bin/bootstrap:
```python
min_bin_count = 20
```
If a bin has fewer than `min_bin_count` observations, mark it unreliable.
For categorical variables:
```python
rare categories -> "__OTHER__"
missing values -> "__MISSING__"
```
Effective Sample Size Warning
Warn when:
```python
n_obs < 500
n_bootstraps < 50
median validation count per bin < min_bin_count
missing_rate > 0.50
```
Sparse Feature Recommendation
If signal is strong but driven by sparse bins:
```python
recommendation = "sparse_signal_review"
```
---
Handling Noisy Residuals
If many univariate models fail to produce positive validation R²:
Do not raise an error by default.
Instead:
Complete the run.
Report that residual signal appears weak or unstable.
Suggest remedies in warnings.
Warnings:
```python
if median_positive_uplift_rate_across_features < 0.10:
    warn("Most features do not produce positive residual lift. Residuals may be mostly noise, candidate features may not contain signal, or the split procedure may be too unstable.")
```
Suggested remedies in warning text:
reduce model flexibility
increase `min_samples_leaf`
reduce number of bins
use repeated K-fold instead of bootstrap
evaluate grouped/segment splits
check whether base predictions are in-sample
check target noise and outliers
use random forest if XGBoost is unstable
increase data size or aggregate sparse categories
---
Model Selection Guidance
Use XGBoost When
relationships may be nonlinear
thresholds matter
missingness should be handled naturally
feature distributions are irregular
you want fast univariate fits
Use Random Forest When
XGBoost is too unstable
residuals are very noisy
sample size is small
you want smoother bagged estimates
Use Simple Trees / Splines Later When
you want more interpretable functional form recommendations
you want direct translation into GLM/GAM/Bayesian model terms
---
Important Warnings
Do Not Overtrust Gain Importance
Gain-based importance should not be primary because it can be biased toward:
continuous variables
high-cardinality categorical variables
variables with many possible split points
correlated predictors
split artifacts in small data
Use permutation importance for screening and out-of-fold residual R² for univariate ranking.
Base Predictions Should Be OOF
If `base_predictions` are in-sample predictions, residuals may be artificially compressed or distorted.
Warn:
```python
"Residual diagnostics are most reliable when base_predictions are out-of-fold or out-of-sample. In-sample predictions can hide residual signal or create misleading artifacts."
```
Correlated Features
Individual rank may be unstable when correlated features are interchangeable.
Optional future enhancement:
cluster correlated features
report cluster-level residual signal
show top representative feature per cluster
---
Minimum Viable Implementation
Implement in this order:
Add univariate residual lift pipeline.
Add repeated bootstrap/repeated split evaluation.
Add OOF residual R².
Add positive uplift rate.
Add median rank and top-5 rate.
Add Spearman median and percent positive.
Add effect curve construction.
Add effect curve stability.
Add null/shadow feature baseline.
Add final summary table.
Add feature diagnostics plots.
Add optional multivariate screening with permutation importance.
---
Acceptance Criteria
The package should be considered updated when the following works:
1. Basic Fit
```python
finder = ResidualSignalFinder(
    screening_enabled=True,
    screening_top_k=50,
    n_bootstraps=100,
    univariate_model_type="xgboost",
)

finder.fit(X, y=actuals, base_pred=base_predictions)

summary = finder.get_summary()
```
`summary` should include:
```python
feature
mean_oof_r2
positive_uplift_rate
median_rank
top_5_rate
median_spearman
percent_positive_spearman
effect_curve_stability
null_beat_rate
stability_score
credible_stability_score
actionability_score
recommendation
```
2. Feature Diagnostics
```python
finder.plot_feature_diagnostics("feature_name")
```
Should produce plots for:
actual vs base vs corrected prediction
mean residual by bin/category
residual model effect curve
bootstrap spaghetti effect curve
residual distribution by bin/category
absolute residual by bin/category
null comparison
missingness diagnostic if applicable
3. Null Baseline
The package should create and evaluate null/shadow features through the same univariate procedure.
Each real feature should receive:
```python
null_beat_rate
```
4. Stability Interpretation
The package should assign recommendation labels:
```python
strong_residual_signal
include_candidate
respecify_existing_feature
watchlist
unstable_signal
likely_noise
variance_candidate
sparse_signal_review
interaction_candidate
```
5. No Hard Failure on Weak Signal
If residual signal is weak, the package should return a valid summary with warnings, not fail.
---
Recommended Default Summary Sorting
```python
summary.sort_values(
    by=[
        "actionability_score",
        "credible_stability_score",
        "mean_oof_r2",
        "top_5_rate",
    ],
    ascending=[False, False, False, False],
)
```
---
Final Design Principle
The package should not merely identify which features predict residuals.
It should identify:
whether the residual signal generalizes out-of-fold,
whether the feature-residual relationship is stable across resamples,
whether the signal beats null/shadow features,
whether the relationship shape suggests inclusion, nonlinear respecification, interaction modeling, missingness treatment, or variance modeling.
The primary diagnostic should be:
```text
stable out-of-fold residual lift + stable effect curve + null baseline credibility
```
not gain-based importance.