from cv2 import imshow
import matplotlib.pyplot as plt
import numpy as np
from plot_utils import set_equal_aspect_ratio_2D

def generate_data(rng=None, seed=None):
    if rng is None:
        rng = np.random
        rng.seed(seed=seed)
     
    nb_classes = 3
    nb_samples_per_class = 100   
    nb_samples = nb_samples_per_class * nb_classes
    nb_dimensions = 2
    
    X = np.zeros((nb_samples, nb_dimensions))
    y = np.zeros(nb_samples, dtype='uint8')
    for j in range(nb_classes):
        ix = range(nb_samples_per_class*j, nb_samples_per_class*(j+1))
        r = np.linspace(0.0, 1.0, nb_samples_per_class)
        t = np.linspace(j*4, (j+1)*4, nb_samples_per_class)
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
        
    return X, y
    
def plot_data(ax, X, y):
    set_equal_aspect_ratio_2D(ax=ax, xs=X[:,0], ys=X[:,1])
    ax.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.Spectral)
    return ax
  
from linear_classifier import train as ltrain
from nn_classifier import train as ntrain
        
def run():
    X, y = generate_data()
    ax  = plt.figure().gca()
    plot_data(ax, X, y)
    
    plot_f(f=ltrain(X, y), title='linear')
    plot_f(f=ntrain(X, y), title='neural network')
    
def plot_f(f, rx=(-2.0,2.0), ry=(-2.0,2.0), res=(512,512), title=''):
    dx = (rx[1] - rx[0]) / float(res[0])
    dy = (ry[1] - ry[0]) / float(res[1])
    I = np.zeros((res[0], res[1], 3), dtype='uint8')
    for py in range(res[1]):
        y = ry[0] + py * dy
        for px in range(res[0]):
            x = rx[0] + px * dx
            c = np.argmax(f(np.array([x,y])))
            if c == 0: 
                I[py,px,:] = np.array([0,255,255])
            elif c == 1:
                I[py,px,:] = np.array([0,0,255])
            else:
                I[py,px,:] = np.array([255,0,0])
    imshow(title, I)