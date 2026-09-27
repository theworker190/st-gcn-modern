import torch

from stgcn.model import STGCN, initialize_weights


def make_model():
    return STGCN(
        in_channels=3,
        num_class=60,
        graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
        edge_importance_weighting=True,
        dropout=0.5,
    )


def test_model_output_shape_and_original_parameter_names():
    model = make_model().eval()
    output = model(torch.zeros(2, 3, 20, 25, 2))

    assert output.shape == (2, 60)
    state = model.state_dict()
    assert "A" in state
    assert "st_gcn_networks.0.gcn.conv.weight" in state
    assert "st_gcn_networks.9.tcn.3.running_var" in state
    assert "edge_importance.0" in state
    assert "fcn.weight" in state


def test_eval_forward_is_deterministic():
    torch.manual_seed(7)
    model = make_model()
    model.apply(initialize_weights)
    model.eval()
    data = torch.randn(1, 3, 16, 25, 2)

    with torch.no_grad():
        first = model(data)
        second = model(data)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
