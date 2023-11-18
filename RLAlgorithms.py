import numpy as np

class TiledQ():
    def __init__(self, A, ranges, num_tilings, bins, offsets):
        self.tilings = []
        for i in range(num_tilings):
            t = []
            for j in range(len(ranges)):
                t.append(np.linspace(ranges[j,0], ranges[j,1], bins[i,j]+1)[1:-1] + offsets[i,j])
            self.tilings.append(t)

        self.tilings     = np.array(self.tilings)
        self.ranges      = ranges
        self.A           = A
        self.state_sizes = [tuple(len(splits)+1 for splits in tiling) for tiling in self.tilings]
        self.q_tables    = [np.zeros((state_size+(len(self.A),))) for state_size in self.state_sizes]
        pass

    def tile_coding(self, feature):
        codings = []
        for t in self.tilings:
            coding = []
            for i in range(len(feature)):
                coding.append(np.digitize(feature[i], t[i]))
            codings.append(coding)
        return np.array(codings)

    def value(self, s):
        codings = self.tile_coding(s)
        qs = []
        for a_idx in len(self.A):
            q = 0
            for coding, q_table in zip(codings, self.q_tables):
                q += q_table[tuple(coding)+(a_idx,)]
            qs.append(q / len(self.tilings))
        return np.array(qs)



class ESGNStepSARSA():
    def __init__(self, MDP, bins, offsets, D, n, epsilon):
        if not (len(bins) == len(offsets) == D):
            raise ValueError('bins and offsets must both have length D')

        self.MDP = MDP
        self.A = self.MDP.A
        self.q = TiledQ(self.MDP.A, self.MDP.get_feature_ranges(), D, bins, offsets)
        self.alpha = 0.5
        self.epsilon = epsilon
        self.w = np.zeros(D)
        self.n = n

    def next_action(self, s):
        A_card = len(self.A)
        probs = np.zeros(A_card) + self.epsilon / A_card

        q = self.w.dot(self.q.value(s))
        probs[np.argmax(q)] += (1 - self.epsilon)

        return np.random.choice(self.A, 1, p=probs) 

    def run(self):
        while True:
            s0 = self.MDP.s
            a0 = self.next_action(s0)
        pass