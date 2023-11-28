import numpy as np

class ESGNStepSARSA():
    def __init__(self, MDP, num_tilings, num_splits, n, epsilon, gamma, alpha, num_episodes=500):
        self.MDP = MDP
        self.A = self.MDP.A
        self.alpha = 0.5
        self.w = np.zeros(num_tilings * num_splits**2 * len(self.A))
        self.n = n
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.num_tilings = num_tilings
        self.num_splits = num_splits

        self.bins = self.code_tiles()

        self.num_episodes = num_episodes

    def code_tiles(self):
        feature_start, feature_stop = self.MDP.get_feature_ranges().T
        feature_span = feature_stop - feature_start

        feature_steps = (1.0/(self.num_tilings-1)) * feature_span
        bins = feature_steps[:,None]*np.arange(self.num_splits) + feature_start[:,None]
        
        offsets = np.random.rand(self.num_tilings, len(feature_span))
        offsets *= np.tile(feature_span, (self.num_tilings, 1))
        offsets += feature_start - feature_span / self.num_splits / 2
        offsets /= self.num_splits
        offsets = np.tile(offsets.T,(self.num_splits,1,1)).T

        return bins + offsets

    def grad_qhat(self, s, a):
        x = np.zeros((len(self.A), self.num_tilings, self.num_splits, self.num_splits))

        a_idx = self.A.index(a)
        for i in range(self.num_tilings):
            j = np.digitize(s[0],self.bins[i,0]) - 1
            k = np.digitize(s[1],self.bins[i,1]) - 1
            x[a_idx, i, j, k] = 1.

        return x.flatten()

    def qhat(self, s, a=None):
        return self.w.dot(self.grad_qhat(s,a))

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

                if tau == T+1:
                    break