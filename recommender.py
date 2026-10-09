import csv
import logging

import numpy as np

logger = logging.getLogger(__name__)


#
# train_model: learns a Biased Funk SVD model by running SGD over the training ratings
#
def train_model(train_data, n_factors=20, n_epochs=20, lr=0.005, reg=0.02):
    #
    # Because raw user/item IDs from the dataset are sparse and non-contiguous,
    # they are remapped to compact 0-based integer indices before any arrays
    # are allocated, keeping memory usage proportional to actual data size.
    #
    # INPUT  = list of (user_id, item_id, rating, timestamp) tuples
    # OUTPUT = dict containing learned matrices, bias vectors, the global mean,
    #          and the ID-to-index maps needed to generate predictions
    #

    #
    # Step 1: enumerate every unique user and item that appears in the training
    # set, then build a dictionary from each raw ID to its compact index.
    #
    all_user_ids = sorted({row[0] for row in train_data})
    all_item_ids = sorted({row[1] for row in train_data})

    user_index = {uid: idx for idx, uid in enumerate(all_user_ids)}
    item_index = {iid: idx for idx, iid in enumerate(all_item_ids)}

    n_users = len(all_user_ids)
    n_items = len(all_item_ids)

    logger.info(f'Training on {len(train_data)} ratings | {n_users} users | {n_items} items')

    #
    # Step 2: compute the global mean rating mu over all training observations.
    #
    # Centering predictions on mu means that bias terms and latent factors
    # only need to account for deviations from this baseline, which accelerates
    # convergence and keeps parameter magnitudes numerically stable.
    #
    rating_vals = np.array([row[2] for row in train_data], dtype=np.float32)
    mu = float(rating_vals.mean())
    logger.info(f'Global mean rating mu = {mu:.4f}')

    #
    # Step 3: initialise the latent factor matrices and bias vectors.
    #
    # Biased Funk SVD approximates the full rating matrix R as:
    #
    #   R_hat[u, i] = mu + b_u[u] + b_i[i] + P[u, :] . Q[i, :]
    #
    # where
    #   mu            = global mean rating (scalar)
    #   b_u[u]        = user bias: how far user u tends to rate above/below mu
    #   b_i[i]        = item bias: how far item i tends to be rated above/below mu
    #   P[u, :]       = latent preference vector for user u (length n_factors)
    #   Q[i, :]       = latent attribute vector for item i (length n_factors)
    #   P[u,:].Q[i,:] = dot product capturing the user-item affinity
    #
    # Separating bias terms from the latent factors is an important design
    # choice: systematic tendencies such as a critical user or a polarising
    # film are absorbed by b_u and b_i, leaving P and Q free to represent
    # genuine preference structure rather than rating-scale offsets.
    #
    # P and Q are initialised with small Gaussian noise to break symmetry,
    # ensuring SGD can distinguish between latent dimensions from the outset.
    # Bias vectors start at zero because no prior offset information is available.
    #
    rng = np.random.default_rng(seed=42)
    P = rng.normal(0, 0.1, (n_users, n_factors)).astype(np.float32)  # user latent factors
    Q = rng.normal(0, 0.1, (n_items, n_factors)).astype(np.float32)  # item latent factors
    b_u = np.zeros(n_users, dtype=np.float32)                        # user biases
    b_i = np.zeros(n_items, dtype=np.float32)                        # item biases

    #
    # Step 4: prepare integer index arrays for efficient access during training.
    #
    # Converting the training tuples into contiguous numpy arrays allows each
    # sample to be retrieved by position in O(1). The order array is shuffled
    # at the start of each epoch to produce unbiased gradient estimates and
    # prevent the model fitting to the original ordering of records in the file.
    #
    n_ratings = len(train_data)
    order = np.arange(n_ratings)

    user_col = np.array([user_index[row[0]] for row in train_data], dtype=np.int32)
    item_col = np.array([item_index[row[1]] for row in train_data], dtype=np.int32)

    #
    # Step 5: run SGD, processing every training rating once per epoch.
    #
    # For each observed rating r_{u,i} the prediction error (residual) is:
    #
    #   e_{u,i} = r_{u,i} - R_hat[u, i]
    #           = r_{u,i} - ( mu + b_u[u] + b_i[i] + P[u,:] . Q[i,:] )
    #
    # All parameters are updated to minimise the L2-regularised squared-error:
    #
    #   L = sum_{(u,i)} e_{u,i}^2
    #       + reg * ( ||P[u,:]||^2 + ||Q[i,:]||^2 + b_u[u]^2 + b_i[i]^2 )
    #
    # The regularisation term (reg) penalises large parameter magnitudes.
    # Without it, the model can drive factors and biases to arbitrarily large
    # values that fit the training data exactly but generalise poorly to
    # unseen ratings(Overfitting).
    #
    # Differentiating L with respect to each parameter via the chain rule gives:
    #
    #   dL / db_u[u]  = -2 * e_{u,i} + 2 * reg * b_u[u]
    #   dL / db_i[i]  = -2 * e_{u,i} + 2 * reg * b_i[i]
    #   dL / dP[u, k] = -2 * e_{u,i} * Q[i, k] + 2 * reg * P[u, k]
    #   dL / dQ[i, k] = -2 * e_{u,i} * P[u, k] + 2 * reg * Q[i, k]
    #
    # The constant factor of 2 is folded into the learning rate lr, giving
    # the gradient-descent update rules applied below.
    #
    # SGD is used rather than batch gradient descent because the training set
    # contains 20 million ratings; computing the full gradient over all samples
    # before each update would be computationally infeasible at that scale.
    # SGD instead updates parameters after each individual rating, making
    # each epoch tractable while still converging to a good solution.
    #
    # Training runs for 40 epochs; beyond this point validation RMSE stops
    # decreasing and begins to rise, signalling that further updates overfit.
    #
    for epoch in range(n_epochs):
        rng.shuffle(order)
        sq_err_sum = 0.0

        for pos in order:
            u = int(user_col[pos])
            i = int(item_col[pos])
            r = float(rating_vals[pos])

            # compute the current prediction and the resulting error
            pred = mu + b_u[u] + b_i[i] + np.dot(P[u], Q[i])
            err = r - pred
            sq_err_sum += err * err

            # gradient step for the bias terms
            b_u[u] += lr * (err - reg * b_u[u])
            b_i[i] += lr * (err - reg * b_i[i])

            # gradient step for the latent factor rows.
            # P[u] is copied before modification so that the Q[i] update uses
            # the original user vector — if the updated P[u] were used instead,
            # the gradient for Q[i] would be incorrect.
            p_u = P[u].copy()
            P[u] += lr * (err * Q[i] - reg * P[u])
            Q[i] += lr * (err * p_u   - reg * Q[i])

        rmse = np.sqrt(sq_err_sum / n_ratings)
        logger.info(f'Epoch {epoch + 1}/{n_epochs}  train RMSE = {rmse:.4f}')

    #
    # Step 6: pack all learned parameters into a dictionary for inference.
    #
    # Including the index maps in the same structure means predict_ratings
    # can translate raw IDs to array positions without any external state.
    #
    return {
        'mu':         mu,
        'P':          P,
        'Q':          Q,
        'b_u':        b_u,
        'b_i':        b_i,
        'user_index': user_index,
        'item_index': item_index,
    }


#
# predict_ratings: applies the trained model to produce a rating for every test pair
#
def predict_ratings(test_data, model):
    #
    # Each test pair (user_id, item_id) is scored using:
    #
    #   R_hat[u, i] = mu + b_u[u] + b_i[i] + P[u, :] . Q[i, :]
    #
    # with the parameters learned during training.
    #
    # When a user or item was absent from the training set (cold-start),
    # only the components that are available are summed. A known user still
    # contributes b_u, which encodes their general rating tendency and is
    # more accurate than falling back to mu alone. A completely unseen
    # (user, item) pair receives mu as its prediction.
    #
    # INPUT  = list of (user_id, item_id, None, timestamp) tuples; trained model dict
    # OUTPUT = list of (user_id, item_id, predicted_rating, timestamp) tuples
    #

    mu         = model['mu']
    P          = model['P']
    Q          = model['Q']
    b_u        = model['b_u']
    b_i        = model['b_i']
    user_index = model['user_index']
    item_index = model['item_index']

    results = []

    for uid, iid, _, timestamp in test_data:
        u = user_index.get(uid)
        i = item_index.get(iid)

        #
        # Assemble the prediction from whichever components are available.
        # The four additive terms are:
        #   mu        = overall average rating
        #   b_u[u]    = how much this user typically deviates from mu
        #   b_i[i]    = how much this item typically deviates from mu
        #   P[u].Q[i] = personalised user-item affinity from the latent space
        #
        if u is not None and i is not None:
            score = mu + b_u[u] + b_i[i] + np.dot(P[u], Q[i])
        elif u is not None:
            score = mu + b_u[u]
        elif i is not None:
            score = mu + b_i[i]
        else:
            score = mu

        #
        # Round to the nearest half-star and clamp to the valid range [0.5, 5.0].
        # Ratings increment in 0.5, so snapping to multiples of 0.5 aligns
        # predictions with the valid rating values and reduces MAE.
        #
        score = round(round(score * 2) / 2, 1)
        score = max(0.5, min(5.0, score))

        results.append((uid, iid, score, timestamp))

    logger.info(f'Generated {len(results)} predictions')
    return results


#
# write_predictions: serialises a list of prediction tuples to a CSV file
#
def write_predictions(filepath, predictions):
    #
    # Each tuple is written as a single row: user_id, item_id, rating, timestamp.
    #
    # INPUT  = destination file path; list of (user_id, item_id, rating, timestamp) tuples
    # OUTPUT = CSV file written to disk
    #
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        for record in predictions:
            writer.writerow(record)

    logger.info(f'Wrote {len(predictions)} predictions to {filepath}')


#
# load_data: reads a rating CSV into a list of uniform tuples
#
def load_data(filepath):
    #
    # Two file formats are supported:
    #   Training (4 columns): user_id, item_id, rating, timestamp
    #   Test     (3 columns): user_id, item_id, timestamp
    #
    # Test files omit the rating column because those values are withheld for
    # evaluation. Setting the rating field to None for test rows means the
    # rest of the pipeline can treat both formats identically.
    #
    # INPUT  = path to a CSV file on disk
    # OUTPUT = list of (user_id, item_id, rating_or_None, timestamp) tuples
    #
    records = []
    with open(filepath, 'r') as f:
        for row in csv.reader(f):
            uid, iid = int(row[0]), int(row[1])
            if len(row) == 4:
                records.append((uid, iid, float(row[2]), int(row[3])))
            elif len(row) == 3:
                records.append((uid, iid, None, int(row[2])))

    logger.info(f'Parsed {len(records)} records from {filepath}')
    return records


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s %(message)s')
    logger.info('Biased Funk SVD recommender system — COMP3208 submission')

    #
    # Load training and test data from disk.
    # INPUT  = CSV files
    # OUTPUT = lists of (user_id, item_id, rating_or_None, timestamp) tuples
    #
    train_data = load_data('train_20M_withratings.csv')
    test_data  = load_data('test_20M_withoutratings.csv')

    #
    # Fit the model to the training data.
    # INPUT  = training tuples and hyperparameters
    # OUTPUT = dict of learned parameters: P, Q, b_u, b_i, mu, user_index, item_index
    #
    # Hyperparameters were chosen via grid search on a held-out validation split;
    # n_factors=50, lr=0.005, reg=0.02 gave the lowest validation RMSE.
    #
    model = train_model(
        train_data,
        n_factors = 50,    # number of latent dimensions
        n_epochs  = 40,    # validation RMSE plateaus here; more epochs overfit
        lr        = 0.005, # SGD step size
        reg       = 0.02,  # L2 regularisation strength
    )

    #
    # Generate a predicted rating for every (user, item) pair in the test set.
    # INPUT  = test tuples and trained model
    # OUTPUT = list of (user_id, item_id, predicted_rating, timestamp) tuples
    #
    predictions = predict_ratings(test_data, model)

    #
    # Write predictions to the submission file.
    # INPUT  = file path and prediction tuples
    # OUTPUT = submission.csv on disk
    #
    write_predictions('submission.csv', predictions)