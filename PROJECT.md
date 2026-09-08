# P3: Model Selection Regret in Low Signal-to-Noise Temporal Tabular Data

Target venue: ICML 2027. Abstract 16 January 2027, paper **22 January 2027**.
Fallbacks: NeurIPS 2027 Datasets and Benchmarks (around May), then ECML PKDD 2027.

This is the highest value target and the earliest hard deadline.

---

## 1. The observation this starts from

One result keeps recurring across large real-world datasets, and the field keeps
explaining it away.

- On 40 million CPRD primary care records, the simple clinical score QRISK3 still
  differentiates cardiovascular risk better than DeepSurv, DeepHit, and random
  survival forests, particularly for women.
- On MIMIC-IV with more than 250,000 ICU patients, XGBoost on tabular features
  reaches AUROC 0.86 while zero-shot LLMs reach 0.50 to 0.61.
- Across IoT, social media, NLP and security corpora, XGBoost dominates on
  structured data.

The standard reading is *"tabular data is different."* Grinsztajn et al. made
that case at NeurIPS 2022 and it has been the field's working explanation since.

**The alternative explanation nobody has tested is that model selection itself is
failing.** As signal-to-noise falls, a validation set stops being able to
distinguish a genuinely better model from a luckier one. Under that hypothesis,
the observed advantage of simple models is partly an artifact of the selection
procedure, not a property of the hypothesis class. Simple models win because
there are fewer ways for validation to pick the wrong one.

---

## 2. The specific claim we contest

Cai and Ye, **ICML 2025**, "Understanding the Limits of Deep Tabular Methods with
Temporal Shift," show that under temporal distribution shift a **random**
validation split outperforms a temporal-ordered one. Their reasoning: a random
split minimises the time lag between training data and test time while reducing
validation bias.

Their result is correct on their data. Our hypothesis is that **it reverses below
a signal-to-noise threshold**.

The intuition: a random validation split trades bias for variance. It gives you a
validation set drawn from closer to the test distribution (less bias), but it
also leaks information about the temporal structure and, more importantly, it
does not change the fundamental amount of signal available to distinguish
candidates. When SNR is high, the bias reduction dominates and the random split
wins. When noise dominates, validation scores are mostly noise regardless of how
you split, and the split that at least preserves the temporal ordering gives a
less optimistic and better calibrated estimate.

Nobody has tested this because **no temporal tabular benchmark contains data
noisy enough**. A monthly equity panel has an R-squared around 0.005, roughly two
orders of magnitude below anything in Wild-Time, TabReD, or the Cai and Ye suite.

WRDS supplies that missing anchor. That is the entire reason this paper is
available to us and not to the tabular ML community.

---

## 3. What this paper contributes

### 3.1 An empirical law

Selection regret as a function of estimated SNR, shift magnitude, and
candidate-set size. The headline figure is regret against SNR, one line per
selection protocol, showing where the ICML 2025 recommendation stops holding.

The metric:

```
regret = L_test(config chosen by validation) - L_test(oracle-best config in the candidate set)
```

Regret isolates selection from everything else. It is zero when validation picks
the best available candidate, regardless of whether that candidate is any good.
It grows when validation picks badly. It is not confounded by whether the model
family is strong, which is what raw test loss confounds.

### 3.2 A pre-modeling diagnostic

A cheap SNR estimator, computable in minutes before any serious training, that
predicts which selection protocol to use. Candidate approaches: a noise-ceiling
estimate from repeated-measure structure where available, and the gap between
oracle-best and constant-predictor performance on held-out data.

This is the practically useful output. Given a new dataset, compute one statistic
and get a protocol recommendation.

### 3.3 An SNR-calibrated selection rule

The constructive contribution. A one-standard-error or inflated-argmax style rule
whose tolerance is **derived from estimated validation noise** rather than fixed
by convention. Shrink toward simpler candidates in proportion to how noisy the
validation signal actually is.

Success criterion: reduces regret at the low-SNR end without costing anything at
the high-SNR end.

---

## 4. Method

### Datasets: 12 to 15 temporal tabular panels spanning three orders of magnitude in SNR

| Band | Sources |
|---|---|
| High SNR | Wild-Time, TabReD and OpenML temporal subsets |
| Medium | MIMIC-IV tabular via Hugging Face, loan default, energy demand |
| **Extreme low** | **WRDS firm-month panel**, characteristics to next-month return, R-squared around 0.005 |

Every dataset gets both a random split and a temporal split, so protocol effects
are measured within dataset rather than across.

### Models

XGBoost, LightGBM, CatBoost, MLP, FT-Transformer, TabM, TabPFN-v2, plus linear
and constant baselines. Identical tuning budget per family, or the comparison
measures budget rather than architecture.

### Selection protocols

Temporal-ordered validation, random-in-train validation (the Cai and Ye
recommendation), rolling-origin, and blocked-purged cross-validation.

### Seeds

**Minimum five per cell, non-negotiable.** The entire paper is a claim about
variance. A single-seed result cannot distinguish a protocol effect from a lucky
draw, which is the exact error the paper is about.

---

## 5. Hard gates

**Gate 1, stage 3.2.** The SNR estimator must track injected noise. Take a
high-SNR dataset, inject controlled label noise at known levels, confirm the
estimator moves monotonically with it. If the estimator does not measure what we
claim, the x-axis of the headline figure is meaningless.

**Gate 2, stage 3.3.** **Reproduce Cai and Ye's result on their own datasets
before touching ours.** We cannot claim their finding inverts at low SNR if we
cannot first reproduce it at their SNR. Their code is public at
`LAMDA-Tabular/Tabular-Temporal-Shift`. If reproduction fails, the paper's
framing has to change before any experiments run.

**Gate 3, stage 3.4.** Regret differences must exceed across-seed variance before
any protocol ranking is claimed.

---

## 6. How we would know we are wrong

- **No inversion.** If regret ordering across protocols is stable all the way
  down to R-squared 0.005, the central hypothesis is false. Fallback: the work
  still constitutes the first benchmark extending temporal tabular evaluation
  into the extreme low SNR regime, which is a Datasets and Benchmarks paper
  rather than an ICML main-track paper. This is the planned pivot, and the
  decision point is mid-November.
- **The estimator does not predict.** If SNR does not predict which protocol
  wins, contribution 3.2 dies and 3.1 becomes descriptive rather than actionable.
- **The rule does not help.** If the calibrated selection rule fails to reduce
  regret, report it. A negative result on a well motivated fix is still
  informative, though it weakens the paper considerably.

---

## 7. Positioning

Related work has to be engaged directly, not cited in passing. Reviewers in this
area know these papers.

- Cai and Ye, ICML 2025. The result we contest.
- Grinsztajn et al., NeurIPS 2022 Datasets and Benchmarks. The "tabular is
  different" thesis we offer an alternative explanation for.
- Drift-Resilient TabPFN, NeurIPS 2024. Temporal shift handled architecturally.
- *Overtuning in Hyperparameter Optimization* (arXiv 2506.19540). Selection regret
  as a concept, but not under temporal shift and not across an SNR range.
- TabReD, TabFSBench, Wild-Time. The benchmark landscape whose SNR range we
  extend.

The novelty is the intersection: selection regret, under temporal shift, across
three orders of magnitude of SNR, anchored by a dataset the community does not
have.

---

## 8. Compute

Many small fits, minutes each. Gradient boosting runs on the MacBook CPU. Only
FT-Transformer, TabM and TabPFN need GPU, which is what the free Kaggle hours
(about 30 per week) and Lightning AI hours are for. The workload shape, thousands
of short independent jobs, suits Kaggle's 9 hour session limit well.

---

## 9. Stages

See `docs/EXECUTION_PLAN.md` section 6. Summary: 3.1 dataset assembly, 3.2 SNR
estimator, 3.3 reproduce Cai and Ye, 3.4 the sweep, 3.5 the selection rule,
3.6 figures.
