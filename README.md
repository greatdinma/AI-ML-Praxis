# Mitigating Gender Bias in Credit Models with Conditional GANs

Doctoral research (D.Eng., Artificial Intelligence & Machine Learning, The George Washington
University, 2025).

Predictive lending models trained on historical loan data inherit the bias in that data. In the
Bondora peer-to-peer lending dataset, a two-tailed independent samples t-test confirms a
statistically significant difference between mean loan amounts awarded to male and female applicants
(p < 0.001) — and gender appears among the top-20 predictors in every baseline model trained on it.
The disparity gets reproduced as if it were a fact about creditworthiness.

This project asks whether a targeted data intervention — generating synthetic female borrower
records with a Conditional GAN and using them to rebalance the training set — can reduce the model's
reliance on gender without giving up predictive performance.

---

## Results

Five XGBoost models, identical algorithm and evaluation throughout, differing only in training data:

| Model | Training data | R² | RMSE | Cross-validated R² |
|---|---|---|---|---|
| **A** | Imbalanced baseline | 0.5318 | 0.6868 | 0.5292 |
| **B** | Balanced by undersampling males | 0.5066 | 0.7051 | 0.5082 |
| **C** | Balanced by oversampling females | 0.5327 | 0.6862 | 0.5690 |
| **D** | Imbalanced + 14,567 synthetic female records | 0.5231 | 0.6932 | 0.6712 |
| **E** | Gender-balanced + 27,551 synthetic female records | 0.5157 | 0.6985 | **0.6737** |

### Fairness

| Metric | Model A (baseline) | Model E (CGAN-augmented) | Change |
|---|---|---|---|
| SHAP gender attribution gap (F − M) | 0.0385 | 0.0288 | **−25%** |
| RMSE gap between genders (F − M) | 0.0567 | 0.0517 | −9% |
| Mean residual difference (F − M) | −0.0025 | −0.0084 | — |
| Between-group variance | 0.0071 | 0.0076 | — |

### What the numbers say

**Cross-validated R² rose 27%** (0.5292 → 0.6737) while **single-split R² fell slightly**
(0.5318 → 0.5157) and RMSE rose slightly (0.6868 → 0.6985). Single-split R² measures performance on
one arbitrary partition and can flatter or punish a model depending on the split it draws.
Cross-validated R² averages across folds and is the better estimate of generalization. Model A's
high single-split score alongside its low cross-validated score is the signature of a model fitted
to its test split rather than to the problem.

**Rebalancing alone does not fix this.** Undersampling males (Model B) degraded everything —
predictive performance, cross-validated R², and the gender RMSE gap all got worse, because
discarding roughly half the male records threw away information. Oversampling (Model C) held
performance and improved stability but left the SHAP gender gap essentially where the baseline had
it (0.0374 vs 0.0385). Only the CGAN-generated records moved the attribution gap meaningfully, and
the larger synthetic set (Model E) moved it furthest.

**Not every metric agrees.** Mean residual difference and between-group variance are marginally
worse in the augmented models than in the baseline. Fairness is not a single quantity: Models D and E
improve on gender attribution and cross-group error parity while giving up a little on residual
centering. Which metric matters most depends on what the model is used for.

---

## Method

**Conditioning.** A binary gender label (0 female, 1 male) is passed through a learned embedding
layer rather than fed in raw, so the generator learns a dense representation of the conditioning
variable and can model each subgroup's distinct feature patterns.

**Generator (Keras).** 100-dimensional noise vector drawn from a standard normal, concatenated with
the embedded gender label, through dense ReLU layers to a tanh output matched to the scaled feature
space. Started shallow for training stability, then deepened, with batch normalization, dropout, and
a Gaussian noise injection layer near the output — the noise layer to push the generator to explore
rather than collapse onto a few patterns, with its standard deviation treated as a tunable
hyperparameter.

**Discriminator (Keras).** Binary classifier over the feature vector concatenated with the same
gender label, dense ReLU layers with dropout, sigmoid output. Trained with **label smoothing** —
real samples targeted at 0.9 rather than 1.0 — to stop the discriminator becoming confident too
early and starving the generator of gradient (Salimans et al., 2016).

**Training.** Custom Keras training loop, non-saturating generator loss, Adam for both networks at
learning rate 1e-5 with β₁ = 0.5, batch size 64, **35,000 steps**. The discriminator updates first
each step, then its weights are frozen while the generator trains; it is alternately frozen and
unfrozen across training so neither network overpowers the other. Both losses are recorded every
step. Checkpoints saved every 5,000 steps, so training can be inspected and rolled back rather than
judged only at the end.

**Generation and integration.** Synthetic feature vectors are produced from fresh noise paired with
a constant female label, inverse-transformed out of MinMaxScaler space back into real financial
units, and merged with the real training data. Two sets were generated: 14,567 records to match the
real female count (Model D) and 27,551 to bring the female population level with the male one
(Model E).

**Validation of the synthetic data.** Correlation analysis confirms the synthetic records preserve
the feature relationships present in the real data while not correlating with individual real
records. Representation improves without opening a re-identification path back to real borrowers.

**Fairness evaluation.** Four metrics, reported together: mean residual difference by gender, RMSE
by gender and the gap between them, between-group variance, and mean SHAP attribution for the gender
feature by actual gender.

---

## Repository structure

```
Codes/     Notebooks: preprocessing, CGAN training, augmentation, model training, evaluation
Data/      Dataset and preprocessed inputs
Models/    Generator checkpoints (every 5,000 steps to 35,000)
Charts/    Figures — loss curves, distribution comparisons, SHAP and fairness plots
Results/   Evaluation output: performance and fairness metrics across Models A–E
```

## Running it

```bash
git clone https://github.com/greatdinma/AI-ML-Praxis.git
cd AI-ML-Praxis
pip install -r requirements.txt
```

Then run the notebooks in `Codes/` in order:

1. **Preprocessing** — load and clean the Bondora data, encode features, MinMaxScaler
2. **Baseline models** — train Models A, B, C (imbalanced, undersampled, oversampled)
3. **CGAN training** — train the conditional generator, checkpointing to `Models/`
4. **Generation & augmentation** — generate synthetic female records, inverse transform, merge
5. **Augmented models & evaluation** — train Models D and E, evaluate performance and fairness

## Data

Bondora publishes its peer-to-peer loan book publicly. Large data files are excluded from this
repository — see `Data/README.md` for download and preprocessing instructions.

---

## Citation

Akor, R. D. (2025). *Enhancing Loan Accessibility for Female Entrepreneurs in Developing Regions
Using Generative AI*. Doctor of Engineering praxis, School of Engineering and Applied Science,
The George Washington University. Defended November 2025.
