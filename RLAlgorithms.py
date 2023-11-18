import numpy as np

def tiling(range, bins, offset):
    return np.linspace(range[0], range[1], bins+1)[1:-1] + offset

def tilings(ranges, num, bins, offsets):
    ts = []
    for i in range(num):
        t = []
        for j in range(len(ranges)):
            t.append(tiling(ranges[j],bins[i,j],offsets[i,j]))
        ts.append(t)
    return np.array(ts)

def tile_coding(feat, ts):
    n = len(feat)
    codings = []
    for t in ts:
        coding = []
        for i in range(n):
            coding.append(np.digitize(feat[i],t[i]))
        codings.append(coding)
    return np.array(coding)

class ESGNStepSARSA():
    def __init__(self, ECMDP, epsilon, D, n):
        self.MDP = ECMDP
        self.alpha = 0.5
        self.epsilon = epsilon
        self.w = np.zeros(D)
        self.n = n
    
    def qhat(self, s, a, w):

        pass

    def next_action(self, s, ):
            
        pass

    def run(self):
        while True:
            s0 = self.MDP.s
            
        pass