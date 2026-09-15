import math
import numpy as np
from xfl_cloudids.baselines import (
    aggregate_fedavg, baseline_names, fedfomo_positive_improvement_weights,
    scaffold_corrected_gradient, weighted_average_states,
)
from xfl_cloudids.evaluation import classification_metrics, validate_result_record


def _states():
    return [
        {"w": np.array([1.0, 2.0]), "b": np.array([0.0])},
        {"w": np.array([3.0, 4.0]), "b": np.array([2.0])},
    ]


def test_registry_contains_required_baselines():
    names=set(baseline_names())
    assert {"local_only","fedavg","fedprox","scaffold","ditto","fedfomo"}.issubset(names)


def test_weighted_average_states():
    out=weighted_average_states(_states(), [0.25,0.75])
    assert np.allclose(out["w"], [2.5,3.5])
    assert np.allclose(out["b"], [1.5])


def test_fedavg_sample_weighted():
    out=aggregate_fedavg(_states(), [1,3])
    assert np.allclose(out["w"], [2.5,3.5])


def test_scaffold_gradient_correction():
    out=scaffold_corrected_gradient([1.0,2.0],[0.3,0.4],[0.1,0.2])
    assert np.allclose(out,[1.2,2.2])


def test_fedfomo_positive_only_weights():
    out=fedfomo_positive_improvement_weights([1,2,3],[-0.1,0.2,0.3])
    assert out[1] == 0.0
    assert math.isclose(out[2],0.4)
    assert math.isclose(out[3],0.6)


def test_fedfomo_all_nonpositive_means_no_peer_mass():
    out=fedfomo_positive_improvement_weights([1,2],[-0.1,0.0])
    assert out == {1:0.0,2:0.0}


def test_common_metrics_binary_case():
    m=classification_metrics([0,0,1,1],[0,1,1,1],[0,1],benign_label=0)
    assert math.isclose(m["accuracy"],0.75)
    assert math.isclose(m["far"],0.5)
    assert 0 <= m["macro_f1"] <= 1
    assert 0 <= m["worst_class_recall"] <= 1


def test_result_schema_accepts_governed_record():
    record={"method":"fedavg","scenario":"a03","seed":17,"round":5,"receiver":0,"split":"gate","accuracy":0.8,"macro_f1":0.7,"far":0.1,"worst_class_recall":0.5}
    validate_result_record(record)
