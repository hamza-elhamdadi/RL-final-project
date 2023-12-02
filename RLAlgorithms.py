import numpy as np

class SARSAAlg:
    def __init__(self, MDP, M):
        self.alpha   = 0.01
        self.tdr     = np.random.rand(1)[0]
        self.epsilon = 0.01
        self.gamma   = 0.5

        self.MDP = MDP
        self.A = self.MDP.A

        self.M = M
        self.w = np.zeros(1 + len(self.MDP.s)*M)

        # self.num_tilings = num_tilings
        # self.num_splits = num_splits

        # self.bins = self.code_tiles()
        # self.w = np.zeros(num_tilings * num_splits**len(self.MDP.s) * len(self.A))

        self.num_episodes = 500

    def x(self, s, a):
        s = self.MDP.get_normalized_state()

        phi = [1]
        for feature in s:
            for i in range(1, self.M + 1):
                phi.append(np.cos(i*np.pi*feature))
            
        return np.array(phi)
                
    def qhat(self, s, a):
        return self.w.dot(self.x(s,a))

    # def code_tiles(self):
    #     feature_start, feature_stop = self.MDP.get_feature_ranges().T
    #     feature_span = feature_stop - feature_start

    #     feature_steps = (1.0/(self.num_splits-1)) * feature_span
    #     bins = feature_steps[:,None]*np.arange(self.num_splits) + feature_start[:,None]
        
    #     offsets = np.random.rand(self.num_tilings, len(feature_span))
    #     offsets *= np.tile(feature_span / self.num_splits * 1.9, (self.num_tilings, 1))
    #     offsets -= feature_span / self.num_splits * 0.95
    #     offsets = np.tile(offsets.T,(self.num_splits,1,1)).T

    #     return bins + offsets
    

    # def grad_qhat(self, s, a):
    #     shape = [len(self.A), self.num_tilings] + [self.num_splits]*len(s)
    #     x = np.zeros(shape)

    #     a_idx = self.A.index(a)
    #     for i in range(self.num_tilings):
    #         jkl = [a_idx, i]
    #         for j in range(len(s)):
    #             jkl.append(np.digitize(s[j],self.bins[i,j]) - 1)
    #         x[(..., *jkl)] = 1.

    #     return x.flatten()
    
    def next_action(self, s):
        A_card = len(self.A)
        probs = np.zeros(A_card) + self.epsilon / A_card

        max_q, max_a_idx = float('-inf'), 0
        for i, a in enumerate(self.A):
            q = self.qhat(s,a)
            if q > max_q:
                max_q = q
                max_a_idx = i
        probs[max_a_idx] += (1 - self.epsilon) 

        return np.random.choice(self.A, 1, p=probs) 

class ESGNStepSARSA(SARSAAlg):
    def __init__(self, MDP, num_tilings, num_splits, n):
        super().__init__(MDP, num_tilings, num_splits)
        self.n = n

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

class TrueOnlineSARSALambda(SARSAAlg):
    def run(self):
        for _ in range(self.num_episodes):
            self.MDP.reset()
            s = self.MDP.s
            a = self.next_action(s)
            x = self.grad_qhat(s, a)
            z = np.zeros(x.shape)
            Q_old = 0
            while not self.MDP.is_terminal():
                self.MDP.next_state(a)
                R = self.MDP.reward()
                s = self.MDP.s
                print(s[0])
                a = self.next_action(s)

                xp = self.grad_qhat(s, a)
                Q = self.w.dot(x)
                Qp = self.qhat(s, a)
                delta = R + self.gamma*Qp - Q
                z = self.gamma*self.tdr*z + (1 - self.alpha*self.gamma*self.tdr*z.dot(x))*x
                self.w += self.alpha*(delta + Q - Q_old)*z - self.alpha*(Q - Q_old)*x
                Q_old = Qp
                x = xp
            
