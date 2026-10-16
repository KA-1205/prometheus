"""Regression checks for mislabeled prevalence and evaluation-label threshold tuning."""
import numpy as np
import pytest
from sklearn.metrics import precision_score, recall_score, f1_score
from src.eval.harness import evaluate_at_prevalence, full_report


def test_unattainable_prevalence_has_no_measured_metric():
    y = np.array([1] * 2111 + [0] * 2799)
    r = evaluate_at_prevalence(y, np.full(len(y), .5), .05)
    assert r['status'] == 'unavailable'
    assert r['requested_prevalence'] == .05
    assert r['source_prevalence'] == pytest.approx(2111 / 4910)
    assert r['achieved_prevalence'] is None
    assert 'pr_auc' not in r


def test_reachable_prevalence_reports_counts_deterministically():
    y, scores = np.array([1] * 10 + [0] * 1000), np.linspace(0, 1, 1010)
    a = evaluate_at_prevalence(y, scores, .05)
    assert a == evaluate_at_prevalence(y, scores, .05)
    assert (a['n_samples'], a['n_fraud'], a['achieved_prevalence']) == (200, 10, .05)


def test_frozen_threshold_and_independent_precision_recall_f1():
    y, scores = np.array([0, 1, 0, 1]), np.array([.2, .4, .6, .8])
    r = full_report(y, scores)['overall']
    pred = scores >= .5
    assert r['decision_threshold'] == .5
    assert r['precision'] == precision_score(y, pred)
    assert r['recall'] == recall_score(y, pred)
    assert r['f1'] == f1_score(y, pred)
    assert r['cost_at_threshold'] == r['cost_per_prevented_fraud'] == 11
    assert 'optimal_threshold' not in r and 'min_cost' not in r


def test_no_prevention_has_no_cost_per_prevention():
    assert full_report([0, 1], [.1, .2])['overall']['cost_per_prevented_fraud'] is None


@pytest.mark.parametrize('labels,scores', [([], []), ([0, 1], [.2]), ([2], [.2]), ([0], [float('nan')]), ([1], [1.1])])
def test_invalid_inputs(labels, scores):
    with pytest.raises(ValueError): full_report(labels, scores)


@pytest.mark.parametrize('p', [0, 1, -1, float('nan')])
def test_invalid_prevalence(p):
    with pytest.raises(ValueError): evaluate_at_prevalence([0, 1], [.1, .9], p)


def test_single_class_pr_auc_is_unavailable():
    assert full_report([0, 0], [.1, .2])['overall']['pr_auc'] is None
