from collections import OrderedDict

import torch

from stgcn.checkpoint import extract_model_state, normalize_state_dict_keys


def test_normalize_data_parallel_prefix():
    weight = torch.ones(2, 3)
    state = OrderedDict(
        [
            ("module.st_gcn_networks.0.gcn.conv.weight", weight),
            ("fcn.bias", torch.zeros(4)),
        ]
    )

    normalized = normalize_state_dict_keys(state)

    assert list(normalized) == ["st_gcn_networks.0.gcn.conv.weight", "fcn.bias"]
    assert normalized["st_gcn_networks.0.gcn.conv.weight"] is weight


def test_extracts_raw_and_wrapped_state_dicts():
    state = {"module.fcn.weight": torch.ones(1)}

    assert list(extract_model_state(state)) == ["fcn.weight"]
    assert list(extract_model_state({"model_state_dict": state})) == ["fcn.weight"]
    assert list(extract_model_state({"state_dict": state})) == ["fcn.weight"]
