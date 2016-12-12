import numpy as np
from classifier_utils import calculate_accuracy, calculate_data_loss, calculate_regularization_loss, calculate_loss

def train(X, y, nb_iterations=10000, step_size=1e-0, rng=None, seed=None):
    if rng is None:
        rng = np.random
        rng.seed(seed=seed)
    
    nb_samples = X.shape[0]
    nb_hidden = 100
    nb_dimensions = X.shape[1]
    nb_classes = np.unique(y).shape[0]
    
    # Weights [nb_dimensions x nb_hidden], [nb_hidden x nb_classes]
    W1 = 0.01 * rng.randn(nb_dimensions, nb_hidden)
    W2 = 0.01 * rng.randn(nb_hidden, nb_classes)
    # Biasses [1 x nb_hidden], [1 x nb_classes]
    b1 = np.zeros((1, nb_hidden))
    b2 = np.zeros((1, nb_classes))
    # Hyperparameters
    regularization_strength = 1e-3
    
    # Gradient descent loop
    for i in range(nb_iterations):
        
        # Evaluate the class scores [nb_samples x nb_hidden], [nb_samples x nb_classes]
        f_hidden = np.maximum(0, np.dot(X, W1) + b1) # ReLU activation function
        f = np.dot(f_hidden, W2) + b2                # Identity activation function
        
        # Compute the accuracy
        accuracy = calculate_accuracy(y=y, f=f)
        print('Training accuracy: {0:.2f}'.format(accuracy))
        
        # Compute the class probabilities [nb_samples x nb_classes]
        # Compute the loss [1]
        data_loss, p        = calculate_data_loss(y=y, f=f)
        regularization_loss = calculate_regularization_loss([W1, W2], regularization_strength=regularization_strength)
        loss                = calculate_loss(data_loss, regularization_loss)
        print('Iteration {0}: loss {1}'.format(i, loss))
        
        # Compute the gradient on the scores [nb_samples x nb_classes]
        df = p
        df[range(nb_samples), y] -= 1
        df /= nb_samples
        # Backpropagate the gradient to the parameters (W2, b2) -> (W1,b1)
        dW2 = np.dot(f_hidden.T, df) + regularization_strength * W2  # [nb_dimensions x nb_classes]
        db2 = np.sum(df, axis=0, keepdims=True)                      # [1 x nb_classes]
        df_hidden = np.dot(df, W2.T)                                 # [nb_samples x nb_hidden]
        df_hidden[f_hidden <= 0] = 0
        dW1 = np.dot(X.T, df_hidden) + regularization_strength * W1  # [nb_dimensions x nb_hidden]
        db1 = np.sum(df_hidden, axis=0, keepdims=True)               # [1 x nb_hidden]
        
        # Perform parameter update
        W1 -= step_size * dW1
        b1 -= step_size * db1
        W2 -= step_size * dW2
        b2 -= step_size * db2
        
    return (lambda x: np.dot(np.maximum(0, np.dot(x, W1) + b1), W2) + b2)