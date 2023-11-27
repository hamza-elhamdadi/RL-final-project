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

    # @abstractmethod
    # def is_terminal(self):
    #     pass

    @abstractmethod
    def reward(self, s=None, a=None):
        pass

    @abstractmethod
    def run_episode(self, policy):
        pass

    # @abstractmethod
    # def get_feature_ranges(self):
    #     pass

class MountainCar(EpisodicContinuousMDP):
    def __init__(self, A=[-1,0,1], x_bounds=[-1.2,0.6], v_bounds=[-0.07, 0.07]):
        self.s = self.initial_state()
        self.A = A
        self.x_lower, self.x_upper = x_bounds
        self.v_lower, self.v_upper = v_bounds
    
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

    def get_feature_ranges(self):
        return np.array([[self.x_lower,self.x_upper],[self.v_lower,self.v_upper]])




class InvertedPendulum(EpisodicContinuousMDP):
    def __init__(self):
        self.s = self.initial_state()
        self.m = 1.0
        self.l = 1.0
        self.g = 10.0
        self.delta_t = 0.05
        self.theta_upper = np.pi
        self.theta_dot_upper = 8.0
        self.tau_upper = 2.0

    def initial_state(self):
        # random angle in [-pi, pi], random angular velocity in [-1,1].
        return [np.random.uniform(-np.pi, np.pi), np.random.uniform(-1, 1)]

    def reset(self):
        self.s = self.initial_state()
    
    def normalize_theta(self, theta):
        return ((theta + np.pi) % (2 * np.pi)) - np.pi
    
    def next_state(self, tau):
        # action is the torque, tau
        theta, theta_dot = self.s
        if tau > self.tau_upper:
            tau = self.tau_upper
        elif tau < -self.tau_upper:
            tau = -self.tau_upper
        new_theta_dot =  theta_dot + (3 * self.g / (2 * self.l) * np.sin(theta) + 3.0 / (self.m * self.l**2) * tau) * self.delta_t

        if new_theta_dot > self.theta_dot_upper:
            new_theta_dot = self.theta_dot_upper
        elif new_theta_dot < -self.theta_dot_upper:
            new_theta_dot = -self.theta_dot_upper

        new_theta = theta + new_theta_dot*self.delta_t

        self.s = [new_theta, new_theta_dot]

    def reward(self, tau):
        theta, theta_dot = self.s
        theta = self.normalize_theta(theta)
        return -(theta ** 2 + 0.1 * (theta_dot ** 2) + 0.001 * (tau ** 2))
        

    def run_episode(self, policy):
        G = 0
        for _ in range(200):
            action = policy(self.s)
            # added reward first because reward function does not use next state to determine reward
            print(self.reward(action))
            G += self.reward(action)
            self.next_state(action)
        self.reset()
        return G


if __name__ == '__main__':
    # mc = MountainCar()
    # print(mc.run_episode(lambda s: -1 if s[1] < 0 else 1))

    pendulum = InvertedPendulum()
    print(pendulum.run_episode(lambda s : 1))
    