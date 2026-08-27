import numpy as np

class FuzzySet:
    """Base class for fuzzy sets"""
    def __init__(self, name):
        self.name = name
    
    def membership(self, x):
        raise NotImplementedError

class TriangularFuzzySet(FuzzySet):
    """Triangular membership function"""
    def __init__(self, name, a, b, c):
        super().__init__(name)
        self.a = a  # left foot
        self.b = b  # peak
        self.c = c  # right foot
    
    def membership(self, x):
        if x <= self.a or x >= self.c:
            return 0.0
        elif x == self.b:
            return 1.0
        elif self.a < x < self.b:
            return (x - self.a) / (self.b - self.a)
        else:
            return (self.c - x) / (self.c - self.b)

class TrapezoidalFuzzySet(FuzzySet):
    """Trapezoidal membership function"""
    def __init__(self, name, a, b, c, d):
        super().__init__(name)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def membership(self, x):
        if x <= self.a or x >= self.d:
            return 0.0
        elif self.a < x < self.b:
            return (x - self.a) / (self.b - self.a)
        elif self.b <= x <= self.c:
            return 1.0
        else:
            return (self.d - x) / (self.d - self.c)

class GaussianFuzzySet(FuzzySet):
    """Gaussian membership function"""
    def __init__(self, name, mean, sigma):
        super().__init__(name)
        self.mean = mean
        self.sigma = sigma
    
    def membership(self, x):
        return np.exp(-((x - self.mean) ** 2) / (2 * (self.sigma ** 2)))