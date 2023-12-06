import numpy as np
from tqdm import tqdm

class SARSAAlg:
    def __init__(self, MDP, M):
        self.alpha   = 0.001
        self.tdr     = 0.9
        self.epsilon = 0.2

        self.MDP = MDP
        self.A = self.MDP.A

        self.M = M
        self.w = np.zeros((len(self.A), 1 + len(self.MDP.s) * self.M ))

        self.num_episodes = 2000

    def reset(self):
        self.w = np.zeros((len(self.A), 1 + len(self.MDP.s) * self.M ))

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

        q = []
        for a_idx in range(A_card):
            q.append(self.w[a_idx].dot(self.x(s)))

        # print('q when selecting action:',q)
        max_a_idx = np.argmax(q)
        probs[max_a_idx] += (1 - self.epsilon) 

        return np.random.choice(self.A, p=probs)

class ESGNStepSARSA(SARSAAlg):
    def __init__(self, MDP, M, n):
        super().__init__(MDP, M)
        self.n = n

    def run(self):
        Gs = []
        for epnum in range(self.num_episodes):
            self.MDP.reset()

            states = []
            actions = []
            rewards = [None]

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

                    self.w[self.A.index(a)] += (self.alpha/(epnum+1))*(G - self.qhat(states[tau], actions[tau])) * self.x(states[tau])

                if tau == T-1:
                    break
            
                t += 1
            
            self.MDP.reset()
            G = 0
            while not self.MDP.is_terminal():
                G += self.MDP.reward()
                a = self.next_action(self.MDP.s)
                self.MDP.next_state(a)
            Gs.append(G)
            # print(f'episode: {epnum}, G =',G)
        return Gs

class TrueOnlineSARSALambda(SARSAAlg):
    def run(self):
        returns = []
        for epnum in tqdm(range(self.num_episodes)):
            alpha = self.alpha


            self.MDP.reset()
            # initialize s
            s = self.MDP.s
            # choose a, epsilon greedy acc to q(s, ., w)
            a = self.next_action(s)
            # x(s, a)
            x = self.x(s)
            # z = vector of zeroes of length = len(x)
            z = np.zeros(x.shape)
            # Q_old = 0
            Q_old = 0
            # loop for each step of episode
            while not self.MDP.is_terminal():
                # Take action a, observe r, s'
                self.MDP.next_state(a)
                R = self.MDP.reward()
                # print(R)
                s = self.MDP.s
                # print(s[0])
                # choose a, epsilon greedy acc to q(s', ., w)
                ap = self.next_action(s)
                # print('action taken:',self.A.index(a))
                a_idx = self.A.index(a)
                ap_idx = self.A.index(ap)

                xp = self.x(s)
                # Why are we using index of a' and not index of a
                Q = self.w[a_idx].dot(x)
                # print(f'w[{a_idx}]:',self.w[a_idx])
                # print(f'x:',x)
                # print('Q:',Q)
                Qp = self.w[ap_idx].dot(xp)
                # print(f'xp:',xp)
                # print('Qp:',Qp)
                # print('z.dot(x):',z.dot(x))
                # print(Q_old, Q, Qp)
                delta = R + self.MDP.gamma*Qp - Q
                # print(delta)
                
                z += x - alpha * z.dot(x) * x
                z *= self.MDP.gamma * self.tdr
                # z += x - alpha * self.MDP.gamma * self.tdr * z.dot(x) * x
                # print(z)
                update = alpha*(delta + Q - Q_old)*z - alpha*(Q - Q_old)*x
                # print('update:',update)
                self.w[a_idx] += update
                Q_old = Qp
                x = xp
                a = ap

            # print('getting return for current episode')

            self.MDP.reset()
            G = 0
            while not self.MDP.is_terminal():
                G += self.MDP.reward()
                a = self.next_action(self.MDP.s)
                self.MDP.next_state(a)
            
            # print(f'episode: {epnum}, G =',G)
            returns.append(G)

        return returns

            # self.epsilon *= 0.9
