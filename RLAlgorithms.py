import numpy as np
from tqdm import tqdm
from scipy.special import softmax

class SARSAAlg:
    def __init__(self, MDP, M, alpha=0.001, epsilon=0.99999, approach='epsilon-greedy'):
        self.alpha   = alpha
        self.base_epsilon = epsilon
        self.epsilon = epsilon
        self.approach = approach

        self.MDP = MDP
        self.A = self.MDP.A

        self.M = M
        self.w = np.zeros((len(self.A), 1 + len(self.MDP.s) * self.M ))

        self.num_episodes = 2000

    def reset(self):
        self.w = np.zeros((len(self.A), 1 + len(self.MDP.s) * self.M ))
        self.epsilon = self.base_epsilon

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
        if not self.approach in ['epsilon-greedy', 'softmax']:
            raise ValueError(f'Invalid approach: {self.approach}. Approach should be either "epsilon-greedy" or "softmax"')

        A_card = len(self.A)
        probs = np.zeros(A_card)

        q = []
        for a_idx in range(A_card):
            q.append(self.w[a_idx].dot(self.x(s)))
        q = np.array(q)

        if self.approach == 'epsilon-greedy':            
            probs += self.epsilon / A_card
            probs[np.argmax(q)] += (1 - self.epsilon) 
        else:
            probs = softmax(q)

        return np.random.choice(self.A, p=probs)

class ESGNStepSARSA(SARSAAlg):
    def __init__(self, MDP, M, n, alpha=0.001, epsilon=0.99999, approach='epsilon-greedy'):
        super().__init__(MDP, M, alpha, epsilon, approach)
        self.n = n

    def run(self):
        Gs = []
        for epnum in tqdm(range(self.num_episodes)):
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
                    rewards.append(self.MDP.reward())
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
    def __init__(self, MDP, M, tdr, alpha=0.001, epsilon=0.99999, approach='softmax'):
        super().__init__(MDP, M, alpha, epsilon, approach)
        self.tdr = tdr

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
                s = self.MDP.s
                # choose a, epsilon greedy acc to q(s', ., w)
                ap = self.next_action(s)
                a_idx = self.A.index(a)
                ap_idx = self.A.index(ap)

                xp = self.x(s)
                Q = self.w[a_idx].dot(x)
                Qp = self.w[ap_idx].dot(xp)
                delta = R + self.MDP.gamma*Qp - Q
                # print(delta)
                
                discount = self.MDP.gamma * self.tdr
                z = discount * z + (1 - alpha * discount * z.dot(x)) * x
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
            
            if self.epsilon > 0.005:
                if epnum % 50 == 0 and epnum > 0:
                    self.epsilon *= self.epsilon
                    self.epsilon = max(self.epsilon, 0.0001)
                    

            # print(f'episode: {epnum}, G =',G)
            returns.append(G)

        return returns

