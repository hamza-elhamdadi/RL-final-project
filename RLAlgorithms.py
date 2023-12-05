import numpy as np
from tqdm import tqdm

class SARSAAlg:
    def __init__(self, MDP, M):
        self.alpha   = 1
        self.tdr     = 0.9
        self.epsilon = 0.01

        self.MDP = MDP
        self.A = self.MDP.A

        self.M = M
        self.w = np.zeros((len(self.A), 1 + len(self.MDP.s) * M ))

        self.num_episodes = 10000

    def x(self, s):
        s = self.MDP.get_normalized_state(s)
        phi = [1]

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
    def __init__(self, MDP, M, n):
        super().__init__(MDP, M)
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
                        G += self.MDP.gamma**(i-tau-1) * rewards[i]

                    if t + self.n < T:
                        G += self.MDP.gamma**self.n * self.qhat(states[tau+self.n], actions[tau+self.n])

                    self.w += (self.alpha/t)*(G - self.qhat(states[tau], actions[tau])) * self.x(states[tau])

                if tau == T-1:
                    break
            
                t += 1

class TrueOnlineSARSALambda(SARSAAlg):
    def run(self):
        for epnum in tqdm(range(self.num_episodes)):
            alp = self.alpha / (epnum+1)
            self.MDP.reset()
            s = self.MDP.s
            a = self.next_action(s)
            x = self.x(s)
            z = np.zeros(x.shape)
            Q_old = 0
            while not self.MDP.is_terminal():
                self.MDP.next_state(a)
                R = self.MDP.reward()
                s = self.MDP.s
                # print(s[0])
                a = self.next_action(s)
                a_idx = self.A.index(a)

                xp = self.x(s)
                Q = self.w[a_idx].dot(x)
                Qp = self.w[a_idx].dot(xp)
                delta = R + self.MDP.gamma*Qp - Q
                
                z = self.MDP.gamma*self.tdr*z + (1 - alp*self.MDP.gamma*self.tdr*z.dot(x))*x
                self.w[a_idx] += alp*(delta + Q - Q_old)*z - alp*(Q - Q_old)*x
                Q_old = Qp
                x = xp
            
