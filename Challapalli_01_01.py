import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def initialize_weights(layers, X_train, seed):
    weights = []
    for i in range(len(layers)):
        np.random.seed(seed)
        if i == 0:
            w = np.random.randn(layers[i], X_train.shape[0] + 1)
        else:
            w = np.random.randn(layers[i], layers[i-1] + 1)
        weights.append(w)
    return weights

def forward_propagation(X, weights):
    activations = [X]
    for i, w in enumerate(weights):
        activation = sigmoid(np.dot(w, np.vstack([np.ones((1, X.shape[1])), activations[-1]])))
        activations.append(activation)
        X = activation
    return activations

def calculate_error(Y, output):
    return np.mean((Y - output) ** 2)

def calculate_gradient(X, Y, weights, h):
    gradients = []
    for layer_index, w in enumerate(weights):
        gradient = np.zeros_like(w)
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                # Positive Step
                w[i, j] += h
                positive_output = forward_propagation(X, weights)[-1]
                positive_error = calculate_error(Y, positive_output)

                # Negative Step
                w[i, j] -= 2 * h
                negative_output = forward_propagation(X, weights)[-1]
                negative_error = calculate_error(Y, negative_output)

                # Reset weight
                w[i, j] += h

                # Calculate gradient
                gradient[i, j] = (positive_error - negative_error) / (2 * h)
        gradients.append(gradient)
    return gradients

def update_weights(weights, gradients, alpha):
    for i in range(len(weights)):
        weights[i] -= alpha * gradients[i]
    return weights

def multi_layer_nn(X_train, Y_train, X_test, Y_test, layers, alpha, epochs, h=0.00001, seed=2):
    weights = initialize_weights(layers, X_train, seed)
    mse_errors = []

    for epoch in range(epochs):
        activations = forward_propagation(X_train, weights)
        gradients = calculate_gradient(X_train, Y_train, weights, h)
        weights = update_weights(weights, gradients, alpha)
        
        #Error when network is frozen
        test_activations = forward_propagation(X_test, weights)
        mse_error = calculate_error(Y_test, test_activations[-1])
        mse_errors.append(mse_error)
        

    output = forward_propagation(X_test, weights)[-1]

    return [weights, np.array(mse_errors),output]


    