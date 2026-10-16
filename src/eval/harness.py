"""Held-out score reporting; never selects a threshold using evaluation labels."""
import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve, precision_score, recall_score

PREVALENCES = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]


def _validated(y_true, y_score):
    y_true, y_score = np.asarray(y_true), np.asarray(y_score, dtype=float)
    if y_true.ndim != 1 or y_score.ndim != 1 or y_true.shape != y_score.shape:
        raise ValueError('labels and scores must be aligned one-dimensional arrays')
    if not len(y_true) or not np.isin(y_true, [0, 1]).all():
        raise ValueError('labels must be non-empty and binary')
    if not np.isfinite(y_score).all() or ((y_score < 0) | (y_score > 1)).any():
        raise ValueError('scores must be finite probabilities in [0, 1]')
    return y_true.astype(int), y_score


def evaluate_at_prevalence(y_true, y_score, prevalence, *, threshold=0.5):
    """Retain all fraud and subsample normals; report impossible requests unavailable."""
    y_true, y_score = _validated(y_true, y_score)
    if not 0 < prevalence < 1:
        raise ValueError('prevalence must be between 0 and 1')
    fraud, normal = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    needed = int(np.ceil(len(fraud) * (1 - prevalence) / prevalence))
    metadata = {'requested_prevalence': float(prevalence), 'source_prevalence': float(y_true.mean()),
                'available_fraud': len(fraud), 'available_normal': len(normal),
                'sampling_policy': 'retain_all_fraud_subsample_normal_without_replacement'}
    if not len(fraud) or needed > len(normal) or needed < 100:
        return {**metadata, 'status': 'unavailable', 'achieved_prevalence': None,
                'error': 'Insufficient class population under declared sampling policy'}
    sampled = np.random.RandomState(42).choice(normal, needed, replace=False)
    indices = np.concatenate([fraud, sampled])
    return {**_compute_metrics(y_true[indices], y_score[indices], threshold=threshold),
            **metadata, 'status': 'available', 'achieved_prevalence': float(y_true[indices].mean()),
            'sampling_seed': 42}


def _compute_metrics(y_true, y_score, *, threshold=0.5):
    y_true, y_score = _validated(y_true, y_score)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be finite and in [0, 1]')
    pr_auc = None
    if len(np.unique(y_true)) == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = float(auc(recall, precision))
    predicted = (y_score >= threshold).astype(int)
    tp = int(((predicted == 1) & (y_true == 1)).sum())
    fp = int(((predicted == 1) & (y_true == 0)).sum())
    fn = int(((predicted == 0) & (y_true == 1)).sum())
    cost = fp + 10 * fn
    metrics = {
        'pr_auc': pr_auc, 'pr_auc_method': 'trapezoidal_precision_recall_curve',
        'n_samples': len(y_true), 'n_fraud': int(y_true.sum()),
        'achieved_prevalence': float(y_true.mean()), 'decision_threshold': float(threshold),
        'threshold_policy': 'predeclared; select on separate validation data if tuned',
        'precision': float(precision_score(y_true, predicted, zero_division=0)),
        'recall': float(recall_score(y_true, predicted, zero_division=0)),
        'f1': float(f1_score(y_true, predicted, zero_division=0)),
        'true_positives': tp, 'false_positives': fp, 'false_negatives': fn,
        'cost_at_threshold': float(cost), 'cost_assumptions': {'false_positive': 1., 'false_negative': 10.},
        'cost_per_prevented_fraud': float(cost / tp) if tp else None,
    }
    for budget_pct in [1, 2, 5, 10]:
        k = max(1, int(len(y_score) * budget_pct / 100))
        top = np.argsort(y_score, kind='stable')[-k:]
        alerts = np.zeros_like(y_true)
        alerts[top] = 1
        metrics[f'recall_at_{budget_pct}pct'] = round(float(recall_score(y_true, alerts, zero_division=0)), 4)
        metrics[f'precision_at_{budget_pct}pct'] = round(float(precision_score(y_true, alerts, zero_division=0)), 4)
        metrics[f'false_decline_rate_{budget_pct}pct'] = round(int(((alerts == 1) & (y_true == 0)).sum()) / len(y_true), 6)
    return metrics


def multi_prevalence_eval(y_true, y_score, *, threshold=0.5):
    return {str(p): evaluate_at_prevalence(y_true, y_score, p, threshold=threshold) for p in PREVALENCES}


def full_report(y_true, y_score, *, threshold=0.5) -> dict:
    """Caller must supply held-out predictions and a frozen threshold."""
    y_true, y_score = _validated(y_true, y_score)
    return {'multi_prevalence': multi_prevalence_eval(y_true, y_score, threshold=threshold),
            'overall': _compute_metrics(y_true, y_score, threshold=threshold),
            'n_samples': len(y_true), 'n_fraud': int(y_true.sum()),
            'fraud_ratio': round(float(y_true.mean()), 6), 'mean_score': round(float(y_score.mean()), 4)}
