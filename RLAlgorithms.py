import numpy as np

class ESGNStepSARSA():
    def __init__(self, ECMDP, qhat, D, n):
        self.MDP = ECMDP
        self.alpha = 0.5
        self.qhat = qhat
        self.w = np.zeros(D)
        self.n = n
    
    def run(self):
        while True:
            s0 = self.MDP.s
        pass