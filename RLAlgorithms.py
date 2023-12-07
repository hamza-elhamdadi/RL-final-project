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

        self.num_episodes = 5

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
        # for each episode
        # for epnum in tqdm(range(self.num_episodes)):
        for epnum in range(self.num_episodes):
            states = []
            actions = []
            rewards = [0]

            # intialize and store s_0
            self.MDP.reset()
            states.append(self.MDP.s)

            # select and store a_0, epsilon greedy/softmax
            a = self.next_action(self.MDP.s)
            # print(a)
            actions.append(a)

            # T = inf
            T, t = float('inf'), 0
            # loop over t until tau = T-1
            while True:
                # print('t, T:', t, T)
                if t < T:
                    # take action a_t
                    self.MDP.next_state(a)
                    # observe  and store r_t+1
                    rewards.append(self.MDP.reward())
                    # observe and store s_t+1
                    states.append(self.MDP.s)
                    # is s_t+1 is terminal
                    if self.MDP.is_terminal():
                        T = t + 1
                    else:
                        # select a_t+1, epsilon_greedy/softmax
                        actions.append(self.next_action(states[-1]))
                # tau = t - n + 1
                tau = t - self.n + 1
                # print('tau:', tau)
                if tau >= 0:
                    lower = tau + 1
                    upper = min(tau+self.n, T)
                    G = 0
                    for i in range(lower, upper+1):
                        # print(len(rewards), i%(self.n + 1))
                        G += (self.MDP.gamma**(i-tau-1) * rewards[i%(self.n + 1)])
                    # print(lower, upper, G, rewards)
                    if t + self.n < T:
                        idx = (tau+self.n)%(self.n + 1)
                        G += self.MDP.gamma**self.n * self.qhat(states[idx], actions[idx])
                    idx = tau%(self.n + 1)
                    self.w[self.A.index(actions[idx])] += self.alpha*(G - self.qhat(states[idx], actions[idx])) * self.x(states[idx])
                    w = []
                    for a in self.A:
                        w.append(self.w[self.A.index(a)].dot(self.x(states[idx])))
                    print(t,T,tau,np.array(w))
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

