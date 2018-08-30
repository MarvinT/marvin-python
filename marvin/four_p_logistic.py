import numpy as np

def four_param_logistic(p):
    """4p logistic function maker.
    
    Returns a function that accepts x and returns y for
    the 4-parameter logistic defined by p.
    
    The 4p logistic is defined by:
    y = A + (K - A) / (1 + exp(-B*(x-M)))
    
    Args:
        p: an iterable of length 4
            A, K, B, M = p
    
    Returns:
        A function that accepts a numpy array as an argument 
        for x values and returns the y values for the defined 4pl curve.
            
    Marvin Thielk 2016
    mthielk@ucsd.edu
    """
    A, K, B, M = p
    def f(x):
        return A + (K - A) / (1 + np.exp(-B*(x-M)))
    return f

def normalized_four_param_logistic(p):
    A, K, B, M = p
    def f(x):
        return 1. / (1. + np.exp(-B*(x-M)))
    return f