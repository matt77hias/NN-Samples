import numpy as np
from classifier_utils import calculate_accuracy, calculate_data_loss, calculate_regularization_loss, calculate_loss

def train(X, y, nb_iterations=10000, step_size=1e-0, rng=None, seed=None):
    if rng is None:
        rng = np.random
        rng.seed(seed=seed)
    
    nb_samples = X.shape[0]
    nb_dimensions = X.shape[1]
    nb_classes = np.unique(y).shape[0]
    
    # Weights [nb_dimensions x nb_classes]
    W = 0.01 * rng.randn(nb_dimensions, nb_classes)
    # Biasses [1 x nb_classes]
    b = np.zeros((1, nb_classes))
    # Hyperparameters
    regularization_strength = 1e-3
    
    # Gradient descent loop
    for i in range(nb_iterations):
        
        # Evaluate the class scores [nb_samples x nb_classes]
        f = np.dot(X, W) + b
        
        # Compute the accuracy
        accuracy = calculate_accuracy(y=y, f=f)
        print('Training accuracy: {0:.2f}'.format(accuracy))
        
        # Compute the class probabilities [nb_samples x nb_classes]
        # Compute the loss [1]
        data_loss, p        = calculate_data_loss(y=y, f=f)
        regularization_loss = calculate_regularization_loss([W], regularization_strength=regularization_strength)
        loss                = calculate_loss(data_loss, regularization_loss)
        print('Iteration {0}: loss {1}'.format(i, loss))
        
        # Compute the gradient on the scores [nb_samples x nb_classes]
        df = p
        df[range(nb_samples), y] -= 1
        df /= nb_samples
        # Backpropagate the gradient to the parameters (W, b)
        dW = np.dot(X.T, df) + regularization_strength * W  # [nb_dimensions x nb_classes]
        db = np.sum(df, axis=0, keepdims=True)              # [1 x nb_classes]
        # Perform parameter update
        W -= step_size * dW
        b -= step_size * db
        
    return (lambda x: np.dot(x, W) + b)