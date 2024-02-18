The goal of this assignment is to implement a multi-layer neural network model.
The neural network model in this assignment is a multi-layer of neurons with multiple nodes in each layer.
The activation (transfer) function of all the nodes is assumed to be a sigmoid function.
Your model weights should include the bias(es).
Your code should be vectorized using numpy. DO NOT use any other package or library (other than numpy).
DO NOT alter/change the name of the function or the parameters of the function. You may introduce additional functions (helper functions) as needed. All the helper functions should be put in the same file with multi_layer_nn()  function
The comments in the multi_layer_nn()  function provide additional information that should help with your implementation.
The Assignment_01_tests.py file includes a very minimal set of unit tests for the multi_layer_nn.py file. The assignment grade will be based on your code passing these tests (and other additional tests).
You may modify the "Assignment_01_tests.py" to include more tests. You may also add additional tests to help you during development of your code.
DO NOT submit the test files file when submitting your Assignment_01
DO NOT  submit the python environment files (if you used an environment for your project)
You may run these tests using the command:      py.test --verbose Assignment_01_tests.py
 
The following is roughly what your output should look like if all tests pass
 
collected 11 items
 
Assignment_01_tests.py::test_can_fit_data_test PASSED                   [  9%]
Assignment_01_tests.py::test_can_fit_data_test_2d PASSED                [ 18%]
Assignment_01_tests.py::test_check_weight_init PASSED                   [ 27%]
Assignment_01_tests.py::test_large_alpha_test PASSED                    [ 36%]
Assignment_01_tests.py::test_small_alpha_test PASSED                    [ 45%]
Assignment_01_tests.py::test_number_of_nodes_test PASSED                [ 54%]
Assignment_01_tests.py::test_check_output_shape PASSED                  [ 63%]
Assignment_01_tests.py::test_check_output_shape_2d PASSED               [ 72%]
Assignment_01_tests.py::test_check_output_values PASSED                 [ 81%]
Assignment_01_tests.py::test_check_weight_update PASSED                 [ 90%]
Assignment_01_tests.py::test_h_value_used PASSED                        [100%]
 
===================== 11 passed in 6.04s ======================================

Description:
def multi_layer_nn(X_train,Y_train,X_test,Y_test,layers,alpha,epochs,h=0.00001,seed=2):
    # This function creates and trains a multi-layer neural Network
    # X_train: Array of input for training [input_dimensions,nof_train_samples]

    # Y_train: Array of desired outputs for training samples [output_dimensions,nof_train_samples]
    # X_test: Array of input for testing [input_dimensions,nof_test_samples]
    # Y_test: Array of desired outputs for test samples [output_dimensions,nof_test_samples]
    # layers: array of integers representing number of nodes in each layer
    # alpha: learning rate
    # epochs: number of epochs for training.
    # h: step size
    # seed: random number generator seed for initializing the weights.
    # return: This function should return a list containing 3 elements:
        # The first element of the return list should be a list of weight matrices.
        # Each element of the list should be a 2-dimensional numpy array corresponds to the weight matrix of the corresponding layer.

        # The second element should be a one dimensional numpy array of numbers
        # representing the average mse error after each epoch. Each error should
        # be calculated by using the X_test array while the network is frozen.
        # This means that the weights should not be adjusted while calculating the error.

        # The third element should be a two-dimensional numpy array [output_dimensions,nof_test_samples]
        # representing the actual output of network when X_test is used as input.

    # Notes:
    # DO NOT use any other package other than numpy
    # Bias should be included in the weight matrix in the first column.
    # Assume that the activation functions for all the layers are sigmoid.
    # Use MSE to calculate error.
    # Use gradient descent for adjusting the weights.
    # use centered difference approximation to calculate partial derivatives.
    # (f(x + h)-f(x - h))/2*h
    # Reseed the random number generator when initializing weights for each layer.
    # i.e., Initialize the weights for each layer by:
pass