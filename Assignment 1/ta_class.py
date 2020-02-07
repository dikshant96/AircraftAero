import numpy as np

class Colpoint:
    def __init__(self,x,y):
        self.x = x
        self.y = y

class Vpoint:
    def __init__(self,x,y):
        self.x = x
        self.y = y

class Panel:
    def __init__(self,x0,y0,x1,y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        dx = x1 - x0
        dy = y1 - y0
        self.ds = np.sqrt(dx**2 + dy**2)
        self.vpoint = Vpoint(x0 + dx/4., y0 + dy/4.)
        self.cpoint = Colpoint(x0 + 3*dx/4., y0 + 3*dy/4.)
        self.beta, self.n1, self.n2, self.t1, self.t2 = self.get_gradient()

    def get_gradient(self):
        dx = self.x1 - self.x0
        dy = self.y1 - self.y0
        beta = np.arctan2(-dy, dx)
        n1 = np.sin(beta)
        n2 = np.cos(beta)
        t1 = np.cos(beta)
        t2 = -np.sin(beta)
        return beta, n1, n2, t1, t2
