# Mitigating Gender Bias in Credit Models with Conditional GANs

Doctoral research (D.Eng., Artificial Intelligence & Machine Learning, The George Washington University, 2025).

Predictive lending models trained on historical loan data inherit the bias in that data. Where women
have historically received smaller loans, a model learns to predict smaller loans for women — and the
disparity is reproduced as if it were a fact about creditworthiness. This project asks whether a
targeted data intervention can break that link without giving up predictive accuracy.

**Approach:** train a Conditional GAN to generate synthetic borrower records conditioned on gender,
use them to rebalance the training set, and re-fit the downstream loan-amount model on the augmented
data. Then measure what moved — both the fairness gap and the accuracy.

---

## Results

| Metric | Baseline | CGAN-augmented |
|---|---|---|
| Fairness gap (gender disparity in predicted loan amount) | — | **narrowed by 23%+** |
| Gender's rank among model predictors | Top predictor | **No longer a top predictor** |
| Cross-validated R² | — | **+15%** |
| Overall predictive accuracy | — | Maintained |

The headline finding: the fairness improvement did not come at the cost of accuracy. R² improved
alongside it, which is the opposite of the trade-off usually assumed when a fairness constraint is
applied post hoc.


---

## Method

**Generator / discriminator.** Conditional GAN implemented in TensorFlow/Keras, with the gender label
supplied as the conditioning variable so the generator learns a class-conditional distribution over
borrower records rather than one pooled distribution.

**Training.** 35,000 steps on GPU, with the generator checkpointed every 5,000 steps
(`generator_step_5000.keras` … `generator_step_35000.keras`). Sample quality was evaluated at each
checkpoint rather than only at the end, so the selected generator is a measured choice and not just
the last one to finish.

**Evaluation.** Baseline and augmented models are evaluated identically on the Bondora peer-to-peer
lending dataset, on the same held-out splits, with fairness and accuracy measured together rather
than in separate passes. Bondora was selected after reviewing several candidate lending datasets;
it was the one with sufficient gender coverage and loan-outcome detail to support the comparison.

---

## Repository structure

```
Codes/     Notebooks: preprocessing, CGAN training, downstream model fitting, evaluation
Data/      Datasets and preprocessed inputs
Models/    Generator checkpoints (every 5,000 steps to 35,000)
Charts/    Figures — training curves, distribution comparisons, fairness plots
Results/   Evaluation output: metrics tables, per-dataset comparisons
```

## Running it

```bash
git clone https://github.com/greatdinma/AI-ML-Praxis.git
cd AI-ML-Praxis
pip install -r requirements.txt
```

Then run the notebooks in `Codes/` in order:

1. **Preprocessing** — load and clean the lending data, encode features
2. **CGAN training** — train the conditional generator, writing checkpoints to `Models/`
3. **Augmentation & downstream model** — generate synthetic records, rebalance, re-fit
4. **Evaluation** — fairness gap, feature importance, cross-validated R²

> Add a `requirements.txt` (see the setup guide) and rename the notebooks so the order is obvious
> from the filenames — `01_preprocessing.ipynb`, `02_cgan_training.ipynb`, and so on.

## Data

Bondora publishes its peer-to-peer loan book publicly. Large data files are excluded from this
repository — see `Data/README.md` for download and preprocessing instructions.

---

## Citation

Akor, R. D. (2026). *Enhancing Loan Accessibility for Female Entrepreneurs in Developing Regions
Using Generative AI*. Doctor of Engineering praxis, School of Engineering and Applied Science,
The George Washington University. Defended November 2025.
