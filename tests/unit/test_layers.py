import numpy as np
import pytest

from netweaver.layers import LayerDense, LayerDropout, _LayerInput


@pytest.fixture
def test_data_layer():  # uses np.random.seed(42)
    return {
        "n_inputs": 3,
        "n_neurons": 4,
        "expected_weights_value": np.array(
            [
                [0.00496714, -0.00138264, 0.00647689, 0.0152303],
                [-0.00234153, -0.00234137, 0.01579213, 0.00767435],
                [-0.00469474, 0.0054256, -0.00463418, -0.0046573],
            ]
        ),  # Shape (3, 4)
        "expected_weights_shape": (3, 4),
        "expected_biases_value": np.array([[0.0, 0.0, 0.0, 0.0]]),  # Shape (1, 4)
        "expected_biases_shape": (1, 4),
        "inputs": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # Shape (2, 3)
        "expected_output_value": np.array(
            [
                [-0.01380016, 0.01021142, 0.02415861, 0.0166071],
                [-0.02000757, 0.01531618, 0.07706312, 0.07134915],
            ]
        ),  # Shape (2, 4)
        "expected_output_shape": (2, 4),
        "dvalues": np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        ),  # Shape (2, 4)
        "expected_dweights_value": np.array(
            [
                [21.0, 26.0, 31.0, 36.0],
                [27.0, 34.0, 41.0, 48.0],
                [33.0, 42.0, 51.0, 60.0],
            ]
        ),  # Shape (3, 4)
        "expected_dweights_shape": (3, 4),
        "expected_dbiases_value": np.array([[6.0, 8.0, 10.0, 12.0]]),  # Shape (1, 4)
        "expected_dbiases_shape": (1, 4),
        "expected_dinputs_value": np.array(
            [
                [0.08255373, 0.07104952, -0.02637528],
                [0.18372049, 0.14618384, -0.06061776],
            ]
        ),  # Shape (2, 3)
        "expected_dinputs_shape": (2, 3),
        "l1_weights_strength": 0.01,
        "l1_biases_strength": 0.02,
        "l2_weights_strength": 0.03,
        "l2_biases_strength": 0.04,
        "expected_dweights_l1_value": np.array(
            [
                [21.01, 25.99, 31.01, 36.01],
                [26.99, 33.99, 41.01, 48.01],
                [32.99, 42.01, 50.99, 59.99],
            ]
        ),
        "expected_dbiases_l1_value": np.array([[6.02, 8.02, 10.02, 12.02]]),  # biases is [0, 0, 0, 0], d(abs(0)) = 1
        "expected_dweights_l2_value": np.array(
            [
                [21.00029803, 25.99991704, 31.00038861, 36.00091382],
                [26.99985951, 33.99985952, 41.00094753, 48.00046046],
                [32.99971832, 42.00032554, 50.99972195, 59.99972056],
            ]
        ),
        "expected_dbiases_l2_value": np.array([[6.0, 8.0, 10.0, 12.0]]),
        "dropout_rate": 0.3,
        "expected_dropout_output_value": np.array(
            [
                [-0.01971451, 0.0, 0.0, 0.02372443],
                [-0.02858224, 0.02188026, 0.11009017, 0.0],
            ]
        ),
        "expected_dropout_dinputs_value": np.array([[1.42857143, 0.0, 0.0, 5.71428571], [7.14285714, 8.57142857, 10.0, 0.0]]),
    }


### Test cases for LayderDense layer


def test_layer_dense_weights_and_biases_shape(test_data_layer):
    """
    Test to ensure that the weights and biases of LayerDense
    are initialized with the correct shapes.
    """

    # Reset the random seed before creating the LayerDense instance
    rng = np.random.default_rng(42)

    # Create a LayerDense instance
    layer = LayerDense(
        n_inputs=test_data_layer.get("n_inputs"),
        n_neurons=test_data_layer.get("n_neurons"),
        rng=rng,
    )

    # Check the value of weights
    np.testing.assert_array_almost_equal(
        layer.weights,
        test_data_layer.get("expected_weights_value"),
        decimal=6,
        err_msg="The weights are not initialized with the expected values",
    )
    # check the value of biases
    np.testing.assert_array_almost_equal(
        layer.biases,
        test_data_layer.get("expected_biases_value"),
        decimal=6,
        err_msg="The biases are not initialized with the expected values",
    )
    # Check the shape of weights
    assert layer.weights.shape == test_data_layer.get("expected_weights_shape"), (
        f"Expected weights shape to be {test_data_layer.get('expected_weights_shape')}",
        f"but got {layer.weights.shape}",
    )

    # Check the shape of biases
    assert layer.biases.shape == test_data_layer.get("expected_biases_shape"), (
        f"Expected biases shape to be {test_data_layer.get('expected_biases_shape')}, but got {layer.biases.shape}"
    )


def test_layer_dense_forward(test_data_layer):
    """
    Test to ensure that the forward method of LayerDense
    computes the correct output with fixed weights and biases.
    """

    # Reset the random seed before creating the LayerDense instance
    rng = np.random.default_rng(42)

    layer = LayerDense(
        n_inputs=test_data_layer.get("n_inputs"),
        n_neurons=test_data_layer.get("n_neurons"),
        rng=rng,
    )

    # Perform forward pass
    layer.forward(inputs=test_data_layer.get("inputs"), training=True)

    # Check output
    np.testing.assert_array_almost_equal(
        layer.output,
        test_data_layer.get("expected_output_value"),
        decimal=6,
        err_msg="Output does not match the expected values.",
    )

    # Check output shape
    assert layer.output.shape == test_data_layer.get("expected_output_shape"), (
        f"Expected output shape to be {test_data_layer.get('expected_output_shape')}, but got {layer.output.shape}"
    )


def test_layer_dense_backward(test_data_layer):
    """
    Test to ensure that the backward method of LayerDense
    computes the correct gradients with fixed weights and biases.
    """

    # Reset the random seed before creating the LayerDense instance
    rng = np.random.default_rng(42)

    layer = LayerDense(n_inputs=test_data_layer.get("n_inputs"), n_neurons=test_data_layer.get("n_neurons"), rng=rng)

    # Perform forward pass
    layer.forward(inputs=test_data_layer.get("inputs"), training=True)

    # Perform backward pass
    layer.backward(dvalues=test_data_layer.get("dvalues"))

    # # Assertions
    # Check dweights
    np.testing.assert_array_almost_equal(
        layer.dweights,
        test_data_layer.get("expected_dweights_value"),
        decimal=6,
        err_msg="dweights does not match the expected values.",
    )

    assert layer.dweights.shape == test_data_layer.get("expected_dweights_shape"), (
        f"Expected dweights shape to be {test_data_layer.get('expected_dweights_shape')}, but got {layer.dweights.shape}"
    )

    # Check dbiases
    np.testing.assert_array_almost_equal(
        layer.dbiases,
        test_data_layer.get("expected_dbiases_value"),
        decimal=6,
        err_msg="dbiases does not match the expected values.",
    )

    assert layer.dbiases.shape == test_data_layer.get("expected_dbiases_shape"), (
        f"Expected dbiases shape to be {test_data_layer.get('expected_dbiases_shape')}, but got {layer.dbiases.shape}"
    )

    # Check dinputs
    np.testing.assert_array_almost_equal(
        layer.dinputs,
        test_data_layer.get("expected_dinputs_value"),
        decimal=6,
        err_msg="dinputs does not match the expected values.",
    )

    assert layer.dinputs.shape == test_data_layer.get("expected_dinputs_shape"), (
        f"Expected dinputs shape to be {test_data_layer.get('expected_dinputs_shape')}, but got {layer.dinputs.shape}"
    )


def test_layer_dense_l1_regularization(test_data_layer):
    """
    Test to ensure that L1 regularization in LayerDense
    computes the correct dweights and dbiases.
    """

    # Set the random seed for reproducibility
    rng = np.random.default_rng(42)

    # Create a LayerDense instance with L1 regularization parameters
    layer = LayerDense(
        n_inputs=test_data_layer.get("n_inputs"),
        n_neurons=test_data_layer.get("n_neurons"),
        weight_regularizer_l1=test_data_layer.get("l1_weights_strength"),  # L1 regularization for weights
        bias_regularizer_l1=test_data_layer.get("l1_biases_strength"),  # L1 regularization for biases
        rng=rng,
    )

    # Perform forward pass (required before backward pass)
    layer.forward(inputs=test_data_layer.get("inputs"), training=True)

    # Perform backward pass to compute gradients
    layer.backward(dvalues=test_data_layer.get("dvalues"))

    # Check dweights with L1 regularization
    # Compare the computed dweights with the expected dweights for L1 regularization
    np.testing.assert_array_almost_equal(
        layer.dweights,
        test_data_layer.get("expected_dweights_l1_value"),
        decimal=6,
        err_msg="dweights with L1 regularization do not match the expected values.",
    )

    # Check dbiases with L1 regularization
    # Compare the computed dbiases with the expected dbiases for L1 regularization
    np.testing.assert_array_almost_equal(
        layer.dbiases,
        test_data_layer.get("expected_dbiases_l1_value"),
        decimal=6,
        err_msg="dbiases with L1 regularization do not match the expected values.",
    )


def test_layer_dense_l2_regularization(test_data_layer):
    """
    Test to ensure that L2 regularization in LayerDense
    computes the correct dweights and dbiases.
    """

    # Set the random seed for reproducibility
    rng = np.random.default_rng(42)

    # Create a LayerDense instance with L2 regularization parameters
    layer = LayerDense(
        n_inputs=test_data_layer.get("n_inputs"),
        n_neurons=test_data_layer.get("n_neurons"),
        weight_regularizer_l2=test_data_layer.get("l2_weights_strength"),  # L2 regularization for weights
        bias_regularizer_l2=test_data_layer.get("l2_biases_strength"),  # L2 regularization for biases
        rng=rng,
    )

    # Perform forward pass (required before backward pass)
    layer.forward(inputs=test_data_layer.get("inputs"), training=True)

    # Perform backward pass to compute gradients
    layer.backward(dvalues=test_data_layer.get("dvalues"))

    # Check dweights with L2 regularization
    # Compare the computed dweights with the expected dweights for L2 regularization
    np.testing.assert_array_almost_equal(
        layer.dweights,
        test_data_layer.get("expected_dweights_l2_value"),
        decimal=6,
        err_msg="dweights with L2 regularization do not match the expected values.",
    )

    # Check dbiases with L2 regularization
    # Compare the computed dbiases with the expected dbiases for L2 regularization
    np.testing.assert_array_almost_equal(
        layer.dbiases,
        test_data_layer.get("expected_dbiases_l2_value"),
        decimal=6,
        err_msg="dbiases with L2 regularization do not match the expected values.",
    )


def test_get_parameters(test_data_layer):
    """
    Test to ensure that the get_parameters method of LayerDense
    returns the correct weights and biases.
    """

    # Reset the random seed before creating the LayerDense instance
    rng = np.random.default_rng(42)

    layer = LayerDense(n_inputs=test_data_layer.get("n_inputs"), n_neurons=test_data_layer.get("n_neurons"), rng=rng)

    # Get the parameters
    weights, biases = layer.get_parameters()

    # Assertions
    # Check weights
    np.testing.assert_array_almost_equal(
        weights,
        test_data_layer.get("expected_weights_value"),
        decimal=6,
        err_msg="Weights do not match the expected values.",
    )

    # Check biases
    np.testing.assert_array_almost_equal(
        biases,
        test_data_layer.get("expected_biases_value"),
        decimal=6,
        err_msg="Biases do not match the expected values.",
    )


def test_set_parameters(test_data_layer):
    """
    Test to ensure that the set_parameters method of LayerDense
    sets the correct weights and biases.
    """

    # Reset the random seed before creating the LayerDense instance
    rng = np.random.default_rng(42)

    layer = LayerDense(
        n_inputs=test_data_layer.get("n_inputs"),
        n_neurons=test_data_layer.get("n_neurons"),
        rng=rng,
    )

    # Set the parameters
    layer.set_parameters(
        weights=test_data_layer.get("expected_weights_value"),
        biases=test_data_layer.get("expected_biases_value"),
    )

    # Assertions
    # Check weights
    np.testing.assert_array_almost_equal(
        layer.weights,
        test_data_layer.get("expected_weights_value"),
        decimal=6,
        err_msg="Weights do not match the expected values.",
    )

    # Check biases
    np.testing.assert_array_almost_equal(
        layer.biases,
        test_data_layer.get("expected_biases_value"),
        decimal=6,
        err_msg="Biases do not match the expected values.",
    )


### Test cases for LayderInput layer


def test_layer_input_forward(test_data_layer):
    """
    Test to ensure that the forward method of _LayerInput
    correctly passes the input data as output.
    """

    # Create an instance of _LayerInput
    input_layer = _LayerInput()

    # Perform forward pass with the input data
    input_layer.forward(test_data_layer.get("inputs"), training=True)

    # Check if the output matches the input data
    np.testing.assert_array_equal(
        input_layer.output,
        test_data_layer.get("inputs"),
        err_msg="_LayerInput forward method output does not match the expected input value",
    )


### Test cases for LayderDropout layer


def test_layer_dropout_forward(test_data_layer):
    """
    Test to ensure that the forward method of LayerDropout
    correctly applies dropout to the input data.
    """

    # Set the random seed for reproducibility
    rng = np.random.default_rng(42)

    # Create an instance of LayerDropout with a dropout rate of 0.5
    dropout_layer = LayerDropout(rate=test_data_layer.get("dropout_rate"), rng=rng)

    # Perform forward pass with the input data
    dropout_layer.forward(test_data_layer.get("expected_output_value"), training=True)

    # Check if the output is the same shape as the input data
    np.testing.assert_array_almost_equal(
        dropout_layer.output,
        test_data_layer.get("expected_dropout_output_value"),
        decimal=6,
        err_msg="LayerDropout forward method output does not match the expected value",
    )


def test_layer_dropout_backward(test_data_layer):
    """
    Test to ensure that the forward method of LayerDropout
    correctly applies dropout to the input data.
    """

    # Set the random seed for reproducibility
    rng = np.random.default_rng(42)

    # Create an instance of LayerDropout with a dropout rate of 0.5
    dropout_layer = LayerDropout(rate=test_data_layer.get("dropout_rate"), rng=rng)

    # Perform forward pass with the input data
    dropout_layer.forward(test_data_layer.get("expected_output_value"), training=True)
    # Perform backward pass with the dvalues
    dropout_layer.backward(test_data_layer.get("dvalues"))

    # Check if the output is the same shape as the input data
    np.testing.assert_array_almost_equal(
        dropout_layer.dinputs,
        test_data_layer.get("expected_dropout_dinputs_value"),
        decimal=6,
        err_msg="LayerDropout backward method dinputs does not match the expected value",
    )
