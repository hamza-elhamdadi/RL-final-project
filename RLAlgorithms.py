import numpy as np

class TiledQ():
    def __init__(self, A, ranges, num_tilings, lr, bins, offsets):
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
        self.lr          = lr

    def tile_coding(self, feature):
        codings = []
        for t in self.tilings:
            coding = []
            for i in range(len(feature)):
                coding.append(np.digitize(feature[i], t[i]))
            codings.append(coding)
        return np.array(codings)
    
    def update(self, s, a, G):
        codings = self.tile_coding(s)
        a_idx = self.A.index(a)
        for coding, q_table in zip(codings, self.q_tables):
            q_table[tuple(coding)+(a_idx,)] += self.lr * (G - q_table[tuple(coding)+(a_idx,)])

    def value(self, s, a=None):
        codings = self.tile_coding(s)
        
        qs = []
        for a_idx in len(self.A):
            q = 0
            for coding, q_table in zip(codings, self.q_tables):
                q += q_table[tuple(coding)+(a_idx,)]
            qs.append(q / len(self.tilings))
        
        if a:
            a_idx = self.A.index(a)
            return qs[a_idx]
        
        return np.array(qs)



class ESGNStepSARSA():
    def __init__(self, MDP, bins, offsets, D, n, epsilon, gamma, alpha, num_episodes=500):
        if not (len(bins) == len(offsets) == D):
            raise ValueError('bins and offsets must both have length D')

        self.MDP = MDP
        self.A = self.MDP.A
        self.q = TiledQ(self.MDP.A, self.MDP.get_feature_ranges(), D, alpha, bins, offsets)
        self.alpha = 0.5
        self.w = np.zeros(D)
        self.n = n
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        self.num_episodes = num_episodes

    def grad_qhat(self, s, a):
        return self.q.value(s, a)

    def qhat(self, s, a=None):
        return self.w.dot(self.q.value(s, a))

    def next_action(self, s):
        A_card = len(self.A)
        probs = np.zeros(A_card) + self.epsilon / A_card

        q = self.qhat(s)
        probs[np.argmax(q)] += (1 - self.epsilon)

        return np.random.choice(self.A, 1, p=probs) 

    def run(self):
        for _ in range(self.num_episodes):
            self.MDP.reset()

            states = []
            actions = []
            rewards = []

            states.append(self.MDP.s)

            a = self.next_action(self.MDP.s)
            actions.append(a)

            T, t = float('inf'), 0
            while True:
                if t < T:
                    self.MDP.next_state(a)
                    rewards.append(self.MDP.reward(self.MDP.s, a))
                    states.append(self.MDP.s)

                    if self.MDP.is_terminal():
                        T = t + 1
                    else:
                        a = self.next_action(self.MDP.s)
                        actions.append(a)
                tau = t - self.n + 1
                if tau >= 0:
                    lower = tau + 1
                    upper = min(tau+self.n, T)
                    G = 0
                    for i in range(lower, upper+1):
                        G += self.gamma**(i-tau-1) * rewards[i]
                        if t + self.n < T:
                            G += self.gamma**self.n * self.qhat(states[tau+self.n], actions[tau+self.n])
                        self.w += self.alpha*(G - self.qhat(states[tau], actions[tau])) * self.grad_qhat(states[tau], actions[tau])

                    self.q.update(states[tau], actions[tau], G)

                if tau == T+1:
                    break