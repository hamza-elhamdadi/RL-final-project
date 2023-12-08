from abc import ABC, abstractmethod
import numpy as np

class EpisodicContinuousMDP(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def initial_state(self):
        pass
    
    @abstractmethod
    def next_state(self, a):
        pass

    @abstractmethod
    def is_terminal(self):
        pass

    @abstractmethod
    def reward(self):
        pass

    @abstractmethod
    def run_episode(self, policy):
        pass

    @abstractmethod
    def get_normalized_state(self, s=None):
        pass

    @abstractmethod
    def get_feature_ranges(self):
        pass

class MountainCar(EpisodicContinuousMDP):
    def __init__(self, gamma=0.9):
        self.s = self.initial_state()
        self.A = [-1,0,1]
        self.x_lower, self.x_upper = [-1.2,  0.6 ]
        self.v_lower, self.v_upper = [-0.07, 0.07]

        self.t = 0

        self.gamma = gamma
    
    def reset(self):
        self.s = self.initial_state()
        self.t = 0

    def initial_state(self):
        pos = np.random.uniform(-0.6, -0.4)
        return np.array([pos, 0])

    def next_state(self, a):
        if self.s[0] == self.x_upper:
            self.s[1] = 0
            return

        self.s[1] = self.s[1] + 0.001*a - 0.0025*np.cos(3*self.s[0])
        self.s[0] += self.s[1]

        if self.s[0] < self.x_lower:
            self.s[0] = self.x_lower
            self.s[1] = 0

        if self.s[0] > self.x_upper:
            self.s[0] = self.x_upper
            self.s[1] = 0

        self.t += 1

    def is_terminal(self):
        return (self.s[0] >= self.x_upper-0.1) or (self.t > 200)

    def reward(self):
        return (self.s[0] >= self.x_upper-0.1) - 1

    def run_episode(self, policy):
        G = 0
        while not self.is_terminal():
            self.next_state(policy(self.s))
            G += self.reward()

        self.reset()
        return G

    def get_normalized_state(self, s):
        mins, maxs = self.get_feature_ranges().T
        return (s - mins) / (maxs - mins)

    def get_feature_ranges(self):
        return np.array([[self.x_lower,self.x_upper],[self.v_lower,self.v_upper]])

class CartPole(EpisodicContinuousMDP):
    def __init__(self):
        self.s = self.initial_state()
        self.A = [-10,10]

        self.g = 9.8                  # gravity
        self.mc = 1.0                 # cart's mass
        self.mp = 0.1                 # pole's mass
        self.mt = 1.1                 # total mass
        self.l = 0.5                  # pole's length
        self.tau = 0.02               # time between action's executed by agent

        self.x_lower, self.x_upper        = -2.4, 2.4
        self.v_lower, self.v_upper        = -2.3, 2.3
        self.w_lower, self.w_upper        = -np.pi/15, np.pi/15
        self.wdot_lower, self.wdot_upper  = -3.4, 3.4

        self.t = 0

        self.gamma = 1.0

    def reset(self):
        self.s = self.initial_state()
        self.t = 0

    def initial_state(self):
        return np.array([0,0,0,0], dtype='float64')
    
    def is_terminal(self):
        return self.s[2] < self.w_lower or self.s[2] > self.w_upper or self.s[0] < self.x_lower or self.s[0] > self.x_upper or self.t > 500
    
    def next_state(self, a):
        b =     (a + self.mp*self.l*(self.s[3]**2)*np.sin(self.s[2]))  /  self.mt
        c =     (self.g*np.sin(self.s[2])-np.cos(self.s[2])*b)         /  (self.l*(4/3 - (self.mp*(np.cos(self.s[2])**2)/self.mt)))
        d = b - (self.mp*self.l*c*np.cos(self.s[2]))                   /  self.mt
        
        update = np.array([self.s[1], d, self.s[3], c])
        self.s += self.tau*update

        self.t += 1

    def reward(self):
        return 1

    def run_episode(self, policy):
        G = 0
        while not self.is_terminal():
            self.next_state(policy(self.s))
            G += self.reward()

        self.reset()
        return G

    def get_normalized_state(self, s):
        mins, maxs = self.get_feature_ranges().T
        return (s - mins) / (maxs - mins)

    def get_feature_ranges(self):
        return np.array([[self.x_lower,self.x_upper],[self.v_lower,self.v_upper],[self.w_lower,self.w_upper],[self.wdot_lower,self.wdot_upper]])

if __name__ == '__main__':
    # mc = MountainCar()
    # print(mc.run_episode(lambda s: -1 if s[1] < 0 else 1))
    cartpole = CartPole()
    print(cartpole.run_episode(lambda x: -10))
    