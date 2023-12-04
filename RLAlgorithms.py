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
        self.w = np.zeros((len(self.A), 1 + len(self.MDP.s) * M ))

        self.num_episodes = 500

    def x(self, s):
        s = self.MDP.get_normalized_state(s)
        phi = []

        for feature in s:
            for i in range(1, self.M + 1):
                phi.append(np.cos(i*np.pi*feature))

        return np.array(phi)
                
    def qhat(self, s, a):
        a_idx = self.A.index(a)
        return self.w[a_idx].dot(self.x(s))
    
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

                    self.w += self.alpha*(G - self.qhat(states[tau], actions[tau])) * self.x(states[tau])

                if tau == T+1:
                    break

class TrueOnlineSARSALambda(SARSAAlg):
    def run(self):
        for _ in range(self.num_episodes):
            self.MDP.reset()
            s = self.MDP.s
            a = self.next_action(s)
            x = self.x(s, a)
            z = np.zeros(x.shape)
            Q_old = 0
            while not self.MDP.is_terminal():
                self.MDP.next_state(a)
                R = self.MDP.reward()
                s = self.MDP.s
                print(s[0])
                a = self.next_action(s)

                xp = self.x(s, a)
                Q = self.w.dot(x)
                Qp = self.qhat(s, a)
                delta = R + self.gamma*Qp - Q
                z = self.gamma*self.tdr*z + (1 - self.alpha*self.gamma*self.tdr*z.dot(x))*x
                self.w += self.alpha*(delta + Q - Q_old)*z - self.alpha*(Q - Q_old)*x
                Q_old = Qp
                x = xp
            
