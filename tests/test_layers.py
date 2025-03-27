import numpy as np
from netweaver.layers import LayerDense
import pytest

@pytest.fixture
def test_data():
    test_dict = {"n_inputs": 3,
                 "n_neurons":4,
                 "expected_weights_value": [[ 0.00496714, -0.00138264,  0.00647689,  0.0152303 ], [-0.00234153, -0.00234137,  0.01579213,  0.00767435], [-0.00469474,  0.0054256,  -0.00463418, -0.0046573 ]],
                 "expected_biases_value": [[0., 0., 0., 0.]],
                 "expected_weights_shape": (3, 4),
                 "expected_biases_shape": (1, 4),
                 "inputs":np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), # Shape (2, 3)
                 "expected_output": [[-0.01380016, 0.01021142, 0.02415861, 0.0166071 ], [-0.02000757, 0.01531618, 0.07706312, 0.07134915]],
                 "expected_output_shape": (2, 4)
                 }
    return test_dict

def test_layer_dense_weights_and_biases_shape(test_data):
    """
    Test to ensure that the weights and biases of LayerDense
    are initialized with the correct shapes.
    """

    # Reset the random seed before creating the LayerDense instance
    np.random.seed(42)

    # Create a LayerDense instance
    layer = LayerDense(n_inputs=test_data.get("n_inputs"), n_neurons=test_data.get("n_neurons"))
    
    # Check the value of weights
    np.testing.assert_array_almost_equal(layer.weights, test_data.get("expected_weights_value"), decimal=6,
                               err_msg="The weights are not initialized with the expected values")
    #check the value of biases
    np.testing.assert_array_almost_equal(layer.biases, test_data.get("expected_biases_value"), decimal=6,
                               err_msg="The biases are not initialized with the expected values")
    # Check the shape of weights
    assert layer.weights.shape == test_data.get("expected_weights_shape"), (
        f"Expected weights shape to be {test_data.get("expected_weights_shape")}",
        f"but got {layer.weights.shape}"
    )

    # Check the shape of biases
    assert layer.biases.shape == test_data.get("expected_biases_shape"), (
        f"Expected biases shape to be {test_data.get("expected_biases_shape")}, "
        f"but got {layer.biases.shape}"
    )

def test_layer_dense_forward(test_data):
    """
    Test to ensure that the forward method of LayerDense
    computes the correct output with fixed weights and biases.
    """

    # Reset the random seed before creating the LayerDense instance
    np.random.seed(42)
    layer = LayerDense(n_inputs=test_data.get("n_inputs"), n_neurons=test_data.get("n_neurons"))

    # Perform forward pass
    layer.forward(inputs=test_data.get("inputs"), training=False)

    # Check output
    np.testing.assert_array_almost_equal(
        layer.output, test_data.get("expected_output"), decimal=6,
        err_msg="Output does not match the expected values."
    )

    # Check output shape
    assert layer.output.shape == test_data.get("expected_output_shape"), (
        f"Expected output shape to be {test_data.get("expected_output_shape")}, "
        f"but got {layer.output.shape}"
    )

def test_layer_dense_backward():
    """
    Test to ensure that the backward method of LayerDense
    computes the correct gradients with fixed weights and biases.
    """
    # Fix the random seed for reproducibility
    np.random.seed(42)

    # Inputs
    n_inputs = 3
    n_neurons = 4
    inputs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Shape (2, 3)
    dvalues = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])  # Shape (2, 4)

    # Expected outputs
    expected_weights = 0.01 * np.random.randn(n_inputs, n_neurons)
    expected_biases = np.zeros((1, n_neurons))
    expected_output = np.dot(inputs, expected_weights) + expected_biases
    expected_output_shape = (inputs.shape[0], n_neurons)
    expected_dweights = np.dot(inputs.T, dvalues)
    expected_dbiases = np.sum(dvalues, axis=0, keepdims=True)
    expected_dinputs = np.dot(dvalues, expected_weights.T)

    # Reset the random seed before creating the LayerDense instance
    np.random.seed(42)
    layer = LayerDense(n_inputs=n_inputs, n_neurons=n_neurons)

    # Perform forward pass
    layer.forward(inputs=inputs, training=False)

    # Perform backward pass
    layer.backward(dvalues=dvalues)

    # # Assertions
    # # Check weights
    # np.testing.assert_array_almost_equal(
    #     layer.weights, expected_weights, decimal=6,
    #     err_msg="Weights do not match the expected values."
    # )

    # # Check biases
    # np.testing.assert_array_almost_equal(
    #     layer.biases, expected_biases, decimal=6,
    #     err_msg="Biases do not match the expected values."
    # )

    # # Check output
    # np.testing.assert_array_almost_equal(
    #     layer.output, expected_output, decimal=6,
    #     err_msg="Output does not match the expected values."
    # )

    # # Check output shape
    # assert layer.output.shape == expected_output_shape, (
    #     f"Expected output shape to be {expected_output_shape}, "
    #     f"but got {layer.output.shape}" 
    # )

    # Check dweights
    np.testing.assert_array_almost_equal(
        layer.dweights, expected_dweights, decimal=6,
        err_msg="dweights does not match the expected values."
    )

    # Check dbiases
    np.testing.assert_array_almost_equal(
        layer.dbiases, expected_dbiases, decimal=6,
        err_msg="dbiases does not match the expected values."
    )

    # Check dinputs
    np.testing.assert_array_almost_equal(
        layer.dinputs, expected_dinputs, decimal=6,
        err_msg="dinputs does not match the expected values."
    )
    


def test_get_parameters():
    """
    Test to ensure that the get_parameters method of LayerDense
    returns the correct weights and biases.
    """
    # Fix the random seed for reproducibility
    np.random.seed(42)

    # Inputs
    n_inputs = 3
    n_neurons = 4

    # Expected outputs
    expected_weights = 0.01 * np.random.randn(n_inputs, n_neurons)
    expected_biases = np.zeros((1, n_neurons))

    # Reset the random seed before creating the LayerDense instance
    np.random.seed(42)
    layer = LayerDense(n_inputs=n_inputs, n_neurons=n_neurons)

    # Get the parameters
    weights, biases = layer.get_parameters()

    # Assertions
    # Check weights
    np.testing.assert_array_almost_equal(
        weights, expected_weights, decimal=6,
        err_msg="Weights do not match the expected values."
    )

    # Check biases
    np.testing.assert_array_almost_equal(
        biases, expected_biases, decimal=6,
        err_msg="Biases do not match the expected values."
    )

def test_set_parameters():
    """
    Test to ensure that the set_parameters method of LayerDense
    sets the correct weights and biases.
    """
    # Fix the random seed for reproducibility
    np.random.seed(42)

    # Inputs
    n_inputs = 3
    n_neurons = 4

    # Expected outputs
    expected_weights = 0.01 * np.random.randn(n_inputs, n_neurons)
    expected_biases = np.zeros((1, n_neurons))

    # Reset the random seed before creating the LayerDense instance
    np.random.seed(42)
    layer = LayerDense(n_inputs=n_inputs, n_neurons=n_neurons)

    # Set the parameters
    layer.set_parameters(weights=expected_weights, biases=expected_biases)

    # Assertions
    # Check weights
    np.testing.assert_array_almost_equal(
        layer.weights, expected_weights, decimal=6,
        err_msg="Weights do not match the expected values."
    )

    # Check biases
    np.testing.assert_array_almost_equal(
        layer.biases, expected_biases, decimal=6,
        err_msg="Biases do not match the expected values."
    )


