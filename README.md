# Matrix Factorization Recommender (Biased Funk SVD)

A movie rating prediction system built from scratch in Python and NumPy. It implements **Biased Funk SVD**, the matrix factorization technique popularised by the Netflix Prize, and trains it with stochastic gradient descent on a dataset of **20 million ratings**.

No recommender libraries (Surprise, implicit, LightFM, etc.) are used. The model, the gradient derivations and the training loop are all hand-written.

## Overview

Given a history of `(user, movie, rating)` interactions, the system learns a latent representation of every user and every movie. It then predicts how a user would rate a movie they haven't seen yet. This is the same rating prediction problem behind recommendation engines at Netflix, Amazon and Spotify.

Each rating is modelled as:

```
r̂(u, i) = μ + b_u + b_i + p_u · q_i
```

| Term | Meaning |
|------|---------|
| `μ` | Global mean rating across the whole dataset |
| `b_u` | User bias: how much a user tends to rate above or below average |
| `b_i` | Item bias: how much a movie tends to be rated above or below average |
| `p_u · q_i` | Dot product of learned latent vectors: personalised user–item affinity |

Splitting the bias terms out from the latent factors means systematic effects (a harsh critic, a universally loved film) are absorbed by `b_u` and `b_i`. That leaves the latent space free to capture real taste structure.

## Key Features

- **SGD training from first principles.** L2-regularised squared-error loss, with gradients derived by hand and documented in the source.
- **Scales to 20M ratings.** Sparse, non-contiguous user and item IDs are remapped to compact indices, so memory grows with the data instead of with the ID range.
- **Cold-start handling.** Unseen users or items fall back to whatever components are available (`μ + b_u`, `μ + b_i`, or `μ`) rather than failing.
- **Output post-processing.** Predictions are snapped to the half-star scale and clamped to `[0.5, 5.0]` to match the valid rating values.
- **Reproducible.** Seeded RNG for initialisation and per-epoch shuffling.
- **Tuned hyperparameters.** Selected by grid search on a held-out validation split: 50 latent factors, learning rate 0.005, regularisation 0.02, 40 epochs (the point where validation RMSE stops improving and overfitting begins).

## Tech Stack

- **Python 3**
- **NumPy** for vectorised linear algebra and parameter storage
- Standard library only otherwise (`csv`, `logging`)

## Running It

```bash
pip install numpy
python recommender.py
```

The script expects these files in the working directory:

| File | Format |
|------|--------|
| `train_20M_withratings.csv` | `user_id, item_id, rating, timestamp` |
| `test_20M_withoutratings.csv` | `user_id, item_id, timestamp` |

It writes `submission.csv`, which contains a predicted rating for every test pair. Training progress (per-epoch training RMSE) is logged to the console.

## Code Structure

All logic is in [`recommender.py`](recommender.py):

| Function | Responsibility |
|----------|----------------|
| `load_data` | Parses training and test CSVs into a single tuple format |
| `train_model` | Builds the index maps, initialises parameters and runs SGD |
| `predict_ratings` | Scores test pairs, handles cold-start cases, rounds and clamps |
| `write_predictions` | Writes predictions to CSV |

The source is heavily commented. It walks through the loss function, the gradient derivation for each parameter, and the reasoning behind design decisions such as copying `p_u` before updating `q_i` so both gradients use the pre-update values.

## What This Project Demonstrates

- Turning a machine learning model from its mathematical formulation into working code
- Optimisation fundamentals: loss design, regularisation, learning rates, convergence and overfitting
- Writing data pipelines that hold up at the scale of tens of millions of records
- Clear technical documentation of design trade-offs

## Possible Extensions

- Vectorise or JIT-compile the SGD inner loop (e.g. with Numba) for faster training
- Add SVD++ to use implicit feedback
- Use the rating timestamps to model temporal drift (timeSVD++)
- Add a proper train/validation/test evaluation harness that reports RMSE and MAE

---

