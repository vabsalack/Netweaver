import numpy as np
import pytest

from netweaver.activation_layers import (
    ActivationLinear,
    ActivationReLU,
    ActivationSigmoid,
    ActivationSoftmax,
)


@pytest.fixture
def test_data_activation():
    return {
        "inputs": np.array(
            [
                [-0.01380016, 0.01021142, 0.02415861, 0.0166071],
                [-0.02000757, 0.01531618, 0.07706312, 0.07134915],
            ]
        ),
        "expected_relu_outputs": np.array(
            [
                [0.0, 0.01021142, 0.02415861, 0.0166071],
                [0.0, 0.01531618, 0.07706312, 0.07134915],
            ]
        ),
        "dvalues": np.array(  # assume inputs as dvalues since no of relu outputs same as inputs
            [
                [-0.01380016, 0.01021142, 0.02415861, 0.0166071],
                [-0.02000757, 0.01531618, 0.07706312, 0.07134915],
            ]
        ),
        "expected_relu_dinputs": np.array(
            [
                [0.0, 0.01021142, 0.02415861, 0.0166071],
                [0.0, 0.01531618, 0.07706312, 0.07134915],
            ]
        ),
        "expected_softmax_outputs": np.array(
            [
                [0.24426796, 0.2502042, 0.25371829, 0.25180955],
                [0.23620821, 0.24470109, 0.26028687, 0.25880384],
            ]
        ),
        "expected_softmax_dinputs": np.array(
            [
                [-0.00569034, 0.00017917, 0.00372035, 0.00179082],
                [-0.0135946, -0.00543963, 0.01028582, 0.00874841],
            ]
        ),
        "expected_softmax_predictions_output": np.array([2, 2]),
        "expected_sigmoid_outputs": np.array(
            [
                [0.49655001, 0.50255283, 0.50603936, 0.50415168],
                [0.49499827, 0.50382897, 0.51925625, 0.51782972],
            ]
        ),
        "expected_sigmoid_dinputs": np.array(
            [
                [-0.00344988, 0.00255279, 0.00603877, 0.00415149],
                [-0.00500139, 0.00382882, 0.0192372, 0.01781461],
            ]
        ),
        "expected_sigmoid_predictions_value": np.array([[0, 1, 1, 1], [0, 1, 1, 1]]),
    }


### Test cases for Relu Activation


def test_activation_relu_forward(test_data_activation):
    """
    Test to ensure that the forward method of ActivationReLU
    computes the correct output for the given inputs.
    """

    # Create an instance of the ActivationReLU class
    activation = ActivationReLU()

    # Perform the forward pass with the input data
    # The ReLU activation function sets all negative values to 0
    activation.forward(test_data_activation.get("inputs"), training=True)

    # Assert that the output matches the expected ReLU outputs
    # Compare the computed output with the expected output
    np.testing.assert_array_equal(
        activation.output,
        test_data_activation.get("expected_relu_outputs"),
        err_msg="ReLU forward pass output do not match the expected output",
    )


def test_activation_relu_backward(test_data_activation):
    """
    Test to ensure that the backward method of ActivationReLU
    computes the correct gradient for the given inputs.
    """
    # Create an instance of the ActivationReLU class
    activation = ActivationReLU()

    # Perform the forward pass with the input data
    activation.forward(test_data_activation.get("inputs"), training=True)

    # Perform the backward pass
    activation.backward(test_data_activation.get("dvalues"))

    # Assert that the computed gradients match the expected output
    np.testing.assert_array_equal(
        activation.dinputs,
        test_data_activation.get("expected_relu_dinputs"),
        err_msg="ReLU backward pass dinputs do not match the expected output",
    )


### Test cases for Softmax Activation


def test_activation_softmax_forward(test_data_activation):
    """
    Test to ensure that the forward method of ActivationSoftmax
    computes the correct output for the given inputs.
    """
    # Create an instance of the ActivationSoftmax class
    activation = ActivationSoftmax()

    # Perform the forward pass with the input data
    activation.forward(test_data_activation.get("inputs"), training=True)

    # Assert that the output matches the expected softmax outputs
    np.testing.assert_array_almost_equal(
        activation.output,
        test_data_activation.get("expected_softmax_outputs"),
        decimal=5,
        err_msg="Softmax forward pass output do not match the expected output",
    )


def test_activation_softmax_backward(test_data_activation):
    """
    Test to ensure that the backward method of ActivationSoftmax
    computes the correct gradient for the given inputs.
    """
    # Create an instance of the ActivationSoftmax class
    activation = ActivationSoftmax()

    # Perform the forward pass with the input data
    activation.forward(test_data_activation.get("inputs"), training=True)

    # Perform the backward pass
    activation.backward(test_data_activation.get("dvalues"))

    # Assert that the computed gradients match the expected output
    np.testing.assert_almost_equal(
        activation.dinputs,
        test_data_activation.get("expected_softmax_dinputs"),
        decimal=6,
        err_msg="Softmax backward pass dinputs do not match the expected dinputs",
    )


def test_activation_softmax_predictions(test_data_activation):
    """
    Test to ensure that the predictions method of ActivationSoftmax
    computes the correct predictions for the given inputs.
    """
    # Create an instance of the ActivationSoftmax class
    activation = ActivationSoftmax()

    # Perform the forward pass with the input data
    activation.forward(test_data_activation.get("inputs"), training=True)

    # Get predictions
    predictions = activation.predictions(activation.output)

    # Assert that the predictions match the expected output
    np.testing.assert_array_equal(
        predictions,
        test_data_activation.get("expected_softmax_predictions_output"),
        err_msg="Softmax predictions do not match the expected output",
    )


### Test cases for Sigmoid Activation


def test_activation_sigmoid_forward(test_data_activation):
    """
    Test to ensure that the forward method of ActivationSigmoid
    computes the correct output for the given inputs.
    """
    # Create an instance of the ActivationSigmoid class
    activation = ActivationSigmoid()

    # Perform the forward pass with the input data
    activation.forward(test_data_activation.get("inputs"), training=True)

    # Assert that the output matches the expected sigmoid outputs
    np.testing.assert_array_almost_equal(
        activation.output,
        test_data_activation.get("expected_sigmoid_outputs"),
        decimal=6,
        err_msg="Sigmoid forward pass output do not match the expected output",
    )


def test_activation_sigmoid_backward(test_data_activation):
    """
    Test to ensure that the backward method of ActivationSigmoid
    computes the correct dinputs for the given inputs.
    """
    # Create an instance of the ActivationSigmoid class
    activation = ActivationSigmoid()

    # Perform the forward pass with the input data
    activation.forward(test_data_activation.get("inputs"), training=True)

    # Perform the backward pass
    activation.backward(test_data_activation.get("dvalues"))

    # Assert that the computed gradients match the expected output
    np.testing.assert_almost_equal(
        activation.dinputs,
        test_data_activation.get("expected_sigmoid_dinputs"),
        decimal=6,
        err_msg="Sigmoid backward pass dinputs do not match the expected output",
    )


def test_activation_sigmoid_predictions(test_data_activation):
    """
    Test to ensure that the predictions method of ActivationSigmoid
    computes the correct predictions for the given inputs.
    """
    # Create an instance of the ActivationSigmoid class
    activation = ActivationSigmoid()

    # Perform the forward pass with the input data
    activation.forward(test_data_activation.get("inputs"), training=True)

    # Get predictions
    predictions = activation.predictions(activation.output)

    # Assert that the predictions match the expected output
    np.testing.assert_array_equal(
        predictions,
        test_data_activation.get("expected_sigmoid_predictions_value"),
        err_msg="Sigmoid predictions do not match the expected output",
    )


### Test cases for Linear Activation


def test_activation_linear_forward(test_data_activation):
    """
    Test to ensure that the forward method of ActivationLinear
    computes the correct output for the given inputs.
    """
    # Create an instance of the ActivationLinear class
    activation = ActivationLinear()

    # Perform the forward pass with the input data
    activation.forward(test_data_activation.get("inputs"), training=True)

    # Assert that the output matches the expected linear outputs
    np.testing.assert_array_equal(
        activation.output,
        test_data_activation.get("inputs"),
        err_msg="Linear forward pass output do not match the expected output",
    )


def test_activation_linear_backward(test_data_activation):
    """
    Test to ensure that the backward method of ActivationLinear
    computes the correct dinputs for the given inputs.
    """
    # Create an instance of the ActivationLinear class
    activation = ActivationLinear()

    # Perform the forward pass with the input data
    activation.forward(test_data_activation.get("inputs"), training=True)

    # Perform the backward pass
    activation.backward(test_data_activation.get("dvalues"))

    # Assert that the computed gradients match the expected output
    np.testing.assert_array_equal(
        activation.dinputs,
        test_data_activation.get("dvalues"),
        err_msg="Linear backward pass dinputs do not match the expected output",
    )
