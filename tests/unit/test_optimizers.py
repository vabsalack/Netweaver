import numpy as np
import pytest

from netweaver.layers import LayerDense
from netweaver.optimizers import OptimizerAdam, OptimizerSGD

### Test OptimizerSGD


@pytest.fixture
def test_data_optimizers():
    """
    Fixture to provide test data for optimizer tests.

    This fixture initializes a LayerDense object with predefined weights,
    biases, and gradients. It also provides expected values for weights,
    biases, and learning rates after updates with and without momentum or decay.
    """
    rng = np.random.default_rng(42)

    # Initialize a LayerDense object with 2 inputs and 2 neurons
    layer = LayerDense(n_inputs=2, n_neurons=2, rng=rng)

    # Set initial weights and biases
    layer.weights = np.array([[0.5, -0.5], [0.3, -0.3]])
    layer.biases = np.array([[0.1, -0.1]])

    # Set gradients for weights and biases
    layer.dweights = np.array([[0.2, -0.2], [0.1, -0.1]])
    layer.dbiases = np.array([[0.05, -0.05]])

    # Create a dictionary with test data and expected results
    return {
        "layer_object": layer,
        "sgd_learning_rate": 0.1,  # Learning rate for SGD
        "expected_weights_without_momentum": np.array(
            [
                [0.48, -0.48],  # Expected weights after 1 update without momentum
                [0.29, -0.29],
            ]
        ),
        "expected_biases_without_momentum": np.array(
            [[0.095, -0.095]]  # Expected biases after 1 update without momentum
        ),
        "sgd_momentum": 0.9,  # Momentum value for SGD
        "expected_weights_with_momentum": np.array(
            [
                [0.442, -0.442],  # Expected weights after 2 updates with momentum
                [0.271, -0.271],
            ]
        ),
        "expected_biases_with_momentum": np.array(
            [[0.0855, -0.0855]]  # Expected biases after 2 updates with momentum
        ),
        "sgd_decay": 0.01,  # Decay rate for learning rate
        "expected_learning_rate": 0.1 / (1 + 0.01 * 1),  # Expected learning rate after 2 updates
    }


def test_optimizer_sgd_basic_update(test_data_optimizers):
    """
    Test that OptimizerSGD correctly updates weights and biases
    without momentum or decay.
    """

    # Initialize the optimizer with a fixed learning rate
    optimizer = OptimizerSGD(learning_rate=0.1)

    # Retrieve the LayerDense object from the test data
    layer_object = test_data_optimizers.get("layer_object")

    # Perform the update on weights and biases using the optimizer
    optimizer.update_params(layer_object)

    # Assert that the updated weights match the expected values
    np.testing.assert_array_almost_equal(
        layer_object.weights,
        test_data_optimizers.get("expected_weights_without_momentum"),
        decimal=6,
        err_msg="Weights were not updated correctly by OptimizerSGD.",
    )

    # Assert that the updated biases match the expected values
    np.testing.assert_array_almost_equal(
        layer_object.biases,
        test_data_optimizers.get("expected_biases_without_momentum"),
        decimal=6,
        err_msg="Biases were not updated correctly by OptimizerSGD.",
    )


def test_optimizer_sgd_with_momentum(test_data_optimizers):
    """
    Test that OptimizerSGD correctly applies momentum to weight and bias updates.
    """

    # Initialize the optimizer with a learning rate and momentum
    optimizer = OptimizerSGD(learning_rate=0.1, momentum=0.9)

    # Retrieve the LayerDense object from the test data
    layer_object = test_data_optimizers.get("layer_object")

    # Perform the first update on weights and biases using the optimizer
    optimizer.update_params(layer_object)

    # Assert that the updated weights match the expected values after the first update
    np.testing.assert_array_almost_equal(
        layer_object.weights,
        test_data_optimizers.get("expected_weights_without_momentum"),
        decimal=6,
        err_msg="Weights were not updated correctly by OptimizerSGD with momentum.",
    )

    # Assert that the updated biases match the expected values after the first update
    np.testing.assert_array_almost_equal(
        layer_object.biases,
        test_data_optimizers.get("expected_biases_without_momentum"),
        decimal=6,
        err_msg="Biases were not updated correctly by OptimizerSGD with momentum.",
    )

    # Perform the second update on weights and biases using the optimizer
    optimizer.update_params(layer_object)

    # Assert that the updated weights match the expected values after the second update
    np.testing.assert_array_almost_equal(
        layer_object.weights,
        test_data_optimizers.get("expected_weights_with_momentum"),
        decimal=6,
        err_msg="Weights were not updated correctly by OptimizerSGD with momentum (second update).",
    )

    # Assert that the updated biases match the expected values after the second update
    np.testing.assert_array_almost_equal(
        layer_object.biases,
        test_data_optimizers.get("expected_biases_with_momentum"),
        decimal=6,
        err_msg="Biases were not updated correctly by OptimizerSGD with momentum (second update).",
    )


def test_optimizer_sgd_with_decay(test_data_optimizers):
    """
    Test that OptimizerSGD correctly applies learning rate decay.
    """

    # Initialize the optimizer with a learning rate and decay rate
    optimizer = OptimizerSGD(learning_rate=0.1, decay=0.01)

    # Retrieve the LayerDense object from the test data
    layer_object = test_data_optimizers.get("layer_object")

    # Perform the first update: prepare, update, and finalize the parameters
    optimizer.pre_update_params()  # Prepare for the update (e.g., calculate decayed learning rate)
    optimizer.update_params(layer_object)  # Update weights and biases
    optimizer.post_update_params()  # Finalize the update (e.g., increment iteration count)

    # Perform the second update: repeat the process to simulate multiple updates
    optimizer.pre_update_params()
    optimizer.update_params(layer_object)
    optimizer.post_update_params()

    # Assert that the current learning rate matches the expected decayed learning rate
    assert optimizer.current_learning_rate == test_data_optimizers.get("expected_learning_rate"), (
        f"Learning rate decay did not work as expected. "
        f"Expected {test_data_optimizers.get('expected_learning_rate')}, got {optimizer.current_learning_rate}."
    )


@pytest.fixture
def test_data_optimizeradam():
    """
    Fixture to provide test data for testing the Adam optimizer.

    This fixture initializes a `LayerDense` object with predefined weights, biases,
    and gradients. It also provides expected results for multiple iterations of
    optimization using the Adam optimizer, including learning rates, updated weights,
    and biases.


    Notes:
        - The random seed is set to ensure reproducibility of the test data.
        - The weights, biases, and gradients are manually set to specific values
          for controlled testing."""
    rng = np.random.default_rng(42)  # Set seed for reproducibility

    # Initialize a LayerDense object with 2 inputs and 2 neurons
    layer = LayerDense(n_inputs=2, n_neurons=2, rng=rng)

    # Set initial weights and biases
    layer.weights = np.array([[0.5, -0.5], [0.3, -0.3]])
    layer.biases = np.array([[0.1, -0.1]])

    # Set gradients for weights and biases
    layer.dweights = np.array([[0.2, 0.2], [0.1, -0.1]])
    layer.dbiases = np.array([[0.05, 0.05]])
    return {
        "layer_object": layer,
        "decay": 0.3,
        "learning_rate": 0.01,
        "learning_rate_iteration_0": 0.01,
        "expected_weights_iteration_0": np.array(
            [
                [0.49, -0.51],
                [0.29000001, -0.29000001],
            ]
        ),
        "expected_biases_iteration_0": np.array([[0.09000002, -0.10999998]]),
        "learning_rate_iteration_1": 0.007692307692307692,
        "expected_weights_iteration_1": np.array(
            [
                [0.4823077, -0.5176923],
                [0.28230771, -0.28230771],
            ]
        ),
        "expected_biases_iteration_1": np.array([[0.08230773, -0.11769227]]),
        "learning_rate_iteration_2": 0.00625,
        "expected_weights_iteration_2": np.array(
            [
                [0.4760577, -0.5239423],
                [0.27605772, -0.27605772],
            ]
        ),
        "expected_biases_iteration_2": np.array([[0.07605774, -0.12394226]]),
    }


## Test OptimizerAdam


def test_optimizer_adam_with_decay(test_data_optimizeradam):
    """
    Test that OptimizerAdam correctly updates weights and biases
    with learning rate decay over three iterations.
    """

    # Initialize the optimizer with a learning rate, decay, and default Adam parameters
    optimizer = OptimizerAdam(
        learning_rate=test_data_optimizeradam.get("learning_rate"),
        decay=test_data_optimizeradam.get("decay"),
    )

    # Retrieve the LayerDense object from the test data
    layer_object = test_data_optimizeradam.get("layer_object")

    # Perform the first update: prepare, update, and finalize the parameters
    optimizer.pre_update_params()  # Prepare for the update (e.g., calculate decayed learning rate)
    optimizer.update_params(layer_object)  # Update weights and biases
    optimizer.post_update_params()  # Finalize the update (e.g., increment iteration count)

    # Assert that the current learning rate matches the expected decayed learning rate after iteration 0
    assert optimizer.current_learning_rate == pytest.approx(test_data_optimizeradam.get("learning_rate_iteration_0"), rel=1e-6), (
        f"Learning rate decay did not work as expected after iteration 0. "
        f"Expected {test_data_optimizeradam.get('learning_rate_iteration_0')}, got {optimizer.current_learning_rate}."
    )

    # Assert that the updated weights match the expected values after iteration 0
    np.testing.assert_array_almost_equal(
        layer_object.weights,
        test_data_optimizeradam.get("expected_weights_iteration_0"),
        decimal=6,
        err_msg="Weights were not updated correctly by OptimizerAdam after iteration 0.",
    )

    # Assert that the updated biases match the expected values after iteration 0
    np.testing.assert_array_almost_equal(
        layer_object.biases,
        test_data_optimizeradam.get("expected_biases_iteration_0"),
        decimal=6,
        err_msg="Biases were not updated correctly by OptimizerAdam after iteration 0.",
    )

    # Perform the second update: prepare, update, and finalize the parameters
    optimizer.pre_update_params()  # Prepare for the update
    optimizer.update_params(layer_object)  # Update weights and biases
    optimizer.post_update_params()  # Finalize the update

    # Assert that the current learning rate matches the expected decayed learning rate after iteration 1
    assert optimizer.current_learning_rate == pytest.approx(test_data_optimizeradam.get("learning_rate_iteration_1"), rel=1e-6), (
        f"Learning rate decay did not work as expected after iteration 1. "
        f"Expected {test_data_optimizeradam.get('learning_rate_iteration_1')}, got {optimizer.current_learning_rate}."
    )

    # Assert that the updated weights match the expected values after iteration 1
    np.testing.assert_array_almost_equal(
        layer_object.weights,
        test_data_optimizeradam.get("expected_weights_iteration_1"),
        decimal=6,
        err_msg="Weights were not updated correctly by OptimizerAdam after iteration 1.",
    )

    # Assert that the updated biases match the expected values after iteration 1
    np.testing.assert_array_almost_equal(
        layer_object.biases,
        test_data_optimizeradam.get("expected_biases_iteration_1"),
        decimal=6,
        err_msg="Biases were not updated correctly by OptimizerAdam after iteration 1.",
    )

    # Perform the third update: prepare, update, and finalize the parameters
    optimizer.pre_update_params()  # Prepare for the update
    optimizer.update_params(layer_object)  # Update weights and biases
    optimizer.post_update_params()  # Finalize the update

    # Assert that the current learning rate matches the expected decayed learning rate after iteration 2
    assert optimizer.current_learning_rate == pytest.approx(test_data_optimizeradam.get("learning_rate_iteration_2"), rel=1e-6), (
        f"Learning rate decay did not work as expected after iteration 2. "
        f"Expected {test_data_optimizeradam.get('learning_rate_iteration_2')}, got {optimizer.current_learning_rate}."
    )

    # Assert that the updated weights match the expected values after iteration 2
    np.testing.assert_array_almost_equal(
        layer_object.weights,
        test_data_optimizeradam.get("expected_weights_iteration_2"),
        decimal=6,
        err_msg="Weights were not updated correctly by OptimizerAdam after iteration 2.",
    )

    # Assert that the updated biases match the expected values after iteration 2
    np.testing.assert_array_almost_equal(
        layer_object.biases,
        test_data_optimizeradam.get("expected_biases_iteration_2"),
        decimal=6,
        err_msg="Biases were not updated correctly by OptimizerAdam after iteration 2.",
    )
