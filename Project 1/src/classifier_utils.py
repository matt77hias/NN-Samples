import numpy as np

def calculate_accuracy(y, f):
    y_pr = np.argmax(f, axis=1)
    return np.mean(y_pr == y)
    
def calculate_loss(data_loss, regularization_loss):
    return data_loss + regularization_loss
    
def calculate_data_loss(y, f):
    return calculate_cross_entropy_loss(y=y, f=f)
    
def calculate_cross_entropy_loss(y, f):
    nb_samples = f.shape[0]
    exp_f = np.exp(f)
    p = exp_f / np.sum(exp_f, axis=1, keepdims=True)
    ls = -np.log(p[range(nb_samples), y])
    return np.sum(ls) / nb_samples, p
        
def calculate_regularization_loss(Ws, regularization_strength=1e-3):
    return reduce(lambda acc, x: acc + np.sum(x*x), Ws, 0)