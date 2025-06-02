import numpy as np
import pytest

from netweaver.lossfunctions import (
    LossBinaryCrossentropy,
    LossCategoricalCrossentropy,
    LossMeanAbsoluteError,
    LossMeanSquaredError,
)


@pytest.fixture
def test_data_cc_loss():
    return {
        "softmax_outputs": np.array(
            [
                [0.24426796, 0.2502042, 0.25371829, 0.25180955],
                [0.23620821, 0.24470109, 0.26028687, 0.25880384],
            ]
        ),  # shape (2, 4)
        "cc_one_hot_true_labels": np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
            ]
        ),  # shape (2, 4)
        "cc_sparse_true_labels": np.array([0, 2]),  # shape (2,)
        "expected_cc_outputs": np.array([1.40948946, 1.34597091]),  # shape (2,)
        "expected_cc_loss_mean": 1.3777301847981456,
        "expected_cc_dinputs": np.array(
            [
                [-2.04693239, -0.0, -0.0, -0.0],
                [-0.0, -0.0, -1.92095744, -0.0],
            ]
        ),
    }


def test_categorical_crossentropy_forward(test_data_cc_loss):
    """
    Test the forward method of LossCategoricalCrossentropy to ensure it computes
    the correct loss values for both one-hot encoded and sparse true labels.
    """

    # Create an instance of the LossCategoricalCrossentropy class
    loss_object = LossCategoricalCrossentropy()

    # Start a new pass (reset any internal state if necessary)
    loss_object.new_pass()

    # Perform forward pass with one-hot encoded true labels
    # This computes the categorical cross-entropy loss for each sample
    loss_array_check_one_hot = loss_object.forward(
        y_pred=test_data_cc_loss.get("softmax_outputs"),
        y_true=test_data_cc_loss.get("cc_one_hot_true_labels"),
    )

    # Perform forward pass with sparse true labels
    # This computes the categorical cross-entropy loss for each sample
    loss_array_check_sparse = loss_object.forward(
        y_pred=test_data_cc_loss.get("softmax_outputs"),
        y_true=test_data_cc_loss.get("cc_sparse_true_labels"),
    )

    # Calculate the mean loss over all samples
    # This computes the average categorical cross-entropy loss
    mean_loss = loss_object.calculate(
        test_data_cc_loss.get("softmax_outputs"),
        test_data_cc_loss.get("cc_sparse_true_labels"),
        include_regularization=False,
    )

    # Assert that the computed loss values for one-hot encoded labels match the expected values
    np.testing.assert_array_almost_equal(
        loss_array_check_one_hot,
        test_data_cc_loss.get("expected_cc_outputs"),
        decimal=6,
        err_msg="LossCategorical forward method with one-hot encoded labels does not match the expected vector",
    )

    # Assert that the computed loss values for sparse labels match the expected values
    np.testing.assert_array_almost_equal(
        loss_array_check_sparse,
        test_data_cc_loss.get("expected_cc_outputs"),
        decimal=6,
        err_msg="LossCategorical forward method with sparse encoded labels does not match the expected vector",
    )

    # Assert that the computed mean loss matches the expected mean loss
    np.testing.assert_almost_equal(
        mean_loss,
        test_data_cc_loss.get("expected_cc_loss_mean"),
        decimal=15,
        err_msg="LossCategorical calculate mean_loss output does not match the expected value",
    )


def test_categorical_crossentropy_backward(test_data_cc_loss):
    """
    Test the backward method of LossCategoricalCrossentropy to ensure it computes
    the correct gradients (dinputs) for the given predictions and true labels.
    """

    # Create an instance of the LossCategoricalCrossentropy class
    loss_object = LossCategoricalCrossentropy()

    # Perform the backward pass
    # This computes the gradients of the loss with respect to the inputs (dinputs)
    loss_object.backward(
        dvalues=test_data_cc_loss.get("softmax_outputs"),  # Predicted probabilities
        y_true=test_data_cc_loss.get("cc_sparse_true_labels"),  # Sparse true labels
    )

    # Assert that the computed gradients (dinputs) match the expected values
    np.testing.assert_array_almost_equal(
        loss_object.dinputs,
        test_data_cc_loss.get("expected_cc_dinputs"),
        decimal=6,
        err_msg="LossCategorical_object.dinputs does not match the expected dinputs values",
    )


@pytest.fixture
def test_data_bc_loss():
    return {
        "sigmoid_outputs": np.array(
            [
                [0.49655001, 0.50255283, 0.50603936, 0.50415168],
                [0.49499827, 0.50382897, 0.51925625, 0.51782972],
            ]
        ),
        "bc_true_label": np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ]
        ),
        "expected_bc_outputs": np.array([0.69524076, 0.68981039]),
        "expected_bc_dinputs": np.array(
            [
                [-0.25173698, 0.25128297, -0.24701636, 0.25209322],
                [0.2475239, -0.24810006, 0.26001378, -0.24139209],
            ]
        ),
    }


def test_binary_crossentropy_forward(test_data_bc_loss):
    """
    Test the forward method of LossBinaryCrossentropy to ensure it computes
    the correct binary cross-entropy loss values for the given predictions
    and true labels.
    """

    # Create an instance of the LossBinaryCrossentropy class
    loss_object = LossBinaryCrossentropy()

    # Perform the forward pass
    # This computes the binary cross-entropy loss for each sample
    binary_loss_array = loss_object.forward(
        test_data_bc_loss.get("sigmoid_outputs"),  # Predicted probabilities (sigmoid outputs)
        test_data_bc_loss.get("bc_true_label"),  # True binary labels
    )

    # Assert that the computed binary cross-entropy loss matches the expected values
    np.testing.assert_array_almost_equal(
        binary_loss_array,
        test_data_bc_loss.get("expected_bc_outputs"),
        decimal=6,
        err_msg="LossBinaryCrossentropy forward method loss values do not match the expected values",
    )


def test_binary_crossentropy_backward(test_data_bc_loss):
    """
    Test the backward method of LossBinaryCrossentropy to ensure it computes
    the correct gradients (dinputs) for the given predictions and true labels.
    """

    # Create an instance of the LossBinaryCrossentropy class
    loss_object = LossBinaryCrossentropy()

    # Perform the backward pass
    # This computes the gradients of the binary cross-entropy loss with respect to the inputs (dinputs)
    loss_object.backward(
        test_data_bc_loss.get("sigmoid_outputs"),  # Predicted probabilities (sigmoid outputs)
        test_data_bc_loss.get("bc_true_label"),  # True binary labels
    )

    # Assert that the computed gradients (dinputs) match the expected values
    np.testing.assert_array_almost_equal(
        loss_object.dinputs,
        test_data_bc_loss.get("expected_bc_dinputs"),
        decimal=6,
        err_msg="LossBinaryCrossentropy_object.dinputs does not match the expected values",
    )


@pytest.fixture
def test_data_mac_loss():
    return {
        "y_pred": np.array(
            [
                [2.5, 0.0, 2.1],
                [1.5, 1.0, 2.0],
            ]
        ),
        "y_true": np.array(
            [
                [3.0, -0.5, 2.0],
                [1.0, 1.0, 2.5],
            ]
        ),
        "expected_mae_outputs": np.array([0.36666667, 0.33333333]),
        "expected_mae_dinputs": np.array(
            [
                [-0.16666667, 0.16666667, 0.16666667],
                [0.16666667, -0.0, -0.16666667],
            ]
        ),
    }


def test_meanabsolute_crossentropy_forward(test_data_mac_loss):
    """
    Test the forward method of LossMeanAbsoluteError to ensure it computes
    the correct mean absolute error (MAE) loss values for the given predictions
    and true labels.
    """

    # Create an instance of the LossMeanAbsoluteError class
    loss_object = LossMeanAbsoluteError()

    # Perform the forward pass
    # This computes the mean absolute error loss for each sample
    mae_loss_array = loss_object.forward(
        test_data_mac_loss.get("y_pred"),  # Predicted values
        test_data_mac_loss.get("y_true"),  # True values
    )

    # Assert that the computed MAE loss matches the expected values
    np.testing.assert_array_almost_equal(
        mae_loss_array,
        test_data_mac_loss.get("expected_mae_outputs"),
        decimal=6,
        err_msg="LossMeanAbsoluteError forward method loss values do not match the expected values",
    )


def test_meanabsolute_crossentropy_backward(test_data_mac_loss):
    """
    Test the backward method of LossMeanAbsoluteError to ensure it computes
    the correct gradients (dinputs) for the given predictions and true labels.
    """

    # Create an instance of the LossMeanAbsoluteError class
    loss_object = LossMeanAbsoluteError()

    # Perform the backward pass
    # This computes the gradients of the mean absolute error loss with respect to the inputs (dinputs)
    loss_object.backward(
        dvalues=test_data_mac_loss.get("y_pred"),  # Predicted values
        y_true=test_data_mac_loss.get("y_true"),  # True values
    )

    # Assert that the computed gradients (dinputs) match the expected values
    np.testing.assert_array_almost_equal(
        loss_object.dinputs,
        test_data_mac_loss.get("expected_mae_dinputs"),
        decimal=6,
        err_msg="LossMeanAbsoluteError backward method dinputs do not match the expected values",
    )


@pytest.fixture
def test_data_mse_loss():
    return {
        "y_pred": np.array(
            [
                [2.5, 0.0, 2.1],
                [1.5, 1.0, 2.0],
            ]
        ),
        "y_true": np.array(
            [
                [3.0, -0.5, 2.0],
                [1.0, 1.0, 2.5],
            ]
        ),
        "expected_mse_outputs": np.array([0.17, 0.16666667]),
        "expected_mse_dinputs": np.array(
            [
                [-0.16666667, 0.16666667, 0.03333333],
                [0.16666667, -0.0, -0.16666667],
            ]
        ),
    }


def test_meansquared_crossentropy_forward(test_data_mse_loss):
    """
    Test the forward method of LossMeanSquaredError to ensure it computes
    the correct mean squared error (MSE) loss values for the given predictions
    and true labels.
    """

    # Create an instance of the LossMeanSquaredError class
    loss_object = LossMeanSquaredError()

    # Perform the forward pass
    # This computes the mean squared error loss for each sample
    mse_loss_array = loss_object.forward(
        test_data_mse_loss.get("y_pred"),  # Predicted values
        test_data_mse_loss.get("y_true"),  # True values
    )

    # Assert that the computed MSE loss matches the expected values
    np.testing.assert_array_almost_equal(
        mse_loss_array,
        test_data_mse_loss.get("expected_mse_outputs"),
        decimal=6,
        err_msg="LossMeanSquaredError forward method loss values do not match the expected values",
    )


def test_meansquared_crossentropy_backward(test_data_mse_loss):
    """
    Test the backward method of LossMeanSquaredError to ensure it computes
    the correct gradients (dinputs) for the given predictions and true labels.
    """

    # Create an instance of the LossMeanSquaredError class
    loss_object = LossMeanSquaredError()

    # Perform the backward pass
    # This computes the gradients of the mean squared error loss with respect to the inputs (dinputs)
    loss_object.backward(
        dvalues=test_data_mse_loss.get("y_pred"),  # Predicted values
        y_true=test_data_mse_loss.get("y_true"),  # True values
    )

    # Assert that the computed gradients (dinputs) match the expected values
    np.testing.assert_array_almost_equal(
        loss_object.dinputs,
        test_data_mse_loss.get("expected_mse_dinputs"),
        decimal=6,
        err_msg="LossMeanSquaredError backward method dinputs do not match the expected values",
    )
