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
    def reward(self, s=None, a=None):
        pass

    @abstractmethod
    def run_episode(self, policy):
        pass

class MountainCar(EpisodicContinuousMDP):
    def __init__(self, A=[-1,0,1], x_bounds=[-1.2,0.5]):
        self.s = self.initial_state()
        self.A = A
        self.x_lower, self.x_upper = x_bounds
    
    def reset(self):
        self.s = self.initial_state()

    def initial_state(self):
        return [-0.5, 0]

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

    def is_terminal(self):
        return self.s[0] == 0.5

    def reward(self, s=None, a=None):
        return (self.s[0] == self.x_upper) - 1

    def run_episode(self, policy):
        G = 0
        while not self.is_terminal():
            self.next_state(policy(self.s))
            G += self.reward()

        self.reset()
        return G


# class InvertedPendulum(EpisodicContinuousMDP):
#     def __init__():
#         pass


if __name__ == '__main__':
    mc = MountainCar()
    print(mc.run_episode(lambda s: -1 if s[1] < 0 else 1))
    