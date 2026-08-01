
import numpy as np
import torch as th
import logging


# ---------------------------------------------------------------------------
# Step 1 — Hide a fraction of positive links before training
# ---------------------------------------------------------------------------

def hide_positive_links(
    df: np.ndarray,
    hidden_ratio: float = 0.10,
    seed: int = 0,
) -> tuple:

    if not (0.0 < hidden_ratio < 1.0):
        raise ValueError(
            "hidden_ratio must be in (0, 1). Got: {}".format(hidden_ratio)
        )

    rng = np.random.default_rng(seed)

    pos_coords = np.array(np.where(df == 1)).T     # shape: (n_pos, 2)
    n_pos      = len(pos_coords)
    n_hide     = max(1, int(round(n_pos * hidden_ratio)))

    chosen_idx   = rng.choice(n_pos, size=n_hide, replace=False)
    hidden_links = pos_coords[chosen_idx]           # shape: (n_hide, 2)

    visible_df = df.copy()
    visible_df[hidden_links[:, 0], hidden_links[:, 1]] = 0

    return hidden_links, visible_df


# ---------------------------------------------------------------------------
# Step 2 — Evaluate hidden-link recovery after training is complete
# ---------------------------------------------------------------------------

@th.no_grad()
def evaluate_hidden_link_recovery(
    model,
    g,
    g_llm,
    feature: dict,
    hidden_links: np.ndarray,
    device: th.device,
    logger: logging.Logger = None,
    threshold: float = 0.5,
) -> dict:

    def _log(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    model.eval()

    # ------------------------------------------------------------------ #
    # Full score matrix (sigmoid over logits)                             #
    # ------------------------------------------------------------------ #
    score_matrix = model([g, g_llm], feature)                    # logits
    pred_matrix  = th.sigmoid(score_matrix).cpu().numpy()        # (n_drugs, n_diseases)

    # ------------------------------------------------------------------ #
    # Extract scores ONLY for the hidden positive links                   #
    # ------------------------------------------------------------------ #
    hd_rows       = hidden_links[:, 0]
    hd_cols       = hidden_links[:, 1]
    scores_hidden = pred_matrix[hd_rows, hd_cols]                # shape: (k,)

    n_hidden    = len(scores_hidden)
    n_recovered = int((scores_hidden >= threshold).sum())

    recovery_rate      = n_recovered / n_hidden
    mean_hidden_score  = float(scores_hidden.mean())
    median_hidden_score = float(np.median(scores_hidden))

    report = dict(
        n_hidden             = n_hidden,
        n_recovered          = n_recovered,
        threshold            = threshold,
        recovery_rate        = recovery_rate,
        mean_score_hidden    = mean_hidden_score,
        median_score_hidden  = median_hidden_score,
        scores_hidden        = scores_hidden,
    )

    # ------------------------------------------------------------------ #
    # Print report                                                         #
    # ------------------------------------------------------------------ #
    _log('\n' + '=' * 60)
    _log('HIDDEN LINK RECOVERY EVALUATION')
    _log('=' * 60)
    _log('  Hidden links (positives removed before training) : {:d}'.format(n_hidden))
    _log('  Threshold used for recovery decision             : {:.2f}'.format(threshold))
    _log('  Recovered (score >= threshold)                   : {:d} / {:d}'.format(
        n_recovered, n_hidden))
    _log('  Recovery Rate @ {:.2f}                          : {:.4f}  ({:.1f}%)'.format(
        threshold, recovery_rate, recovery_rate * 100))
    _log('  Mean   score on hidden positives                 : {:.4f}'.format(mean_hidden_score))
    _log('  Median score on hidden positives                 : {:.4f}'.format(median_hidden_score))
    _log('=' * 60 + '\n')

    return report