from MDP import MountainCar
from RLAlgorithms import ESGNStepSARSA, TrueOnlineSARSALambda
import matplotlib.pyplot as plt
import numpy as np

num_trials = 5
mc = MountainCar()

hp_settings = [
    (3, 0.5, 0.001),
    (5, 0.5, 0.001),
    (3, 0.4, 0.001),
    (3, 0.6, 0.001),
    (3, 0.5, 0.01),
    (3, 0.5, 0.005),
    (2, 0.5, 0.001)
]

for h in hp_settings:
    M, tdr, alpha = h

    alg = TrueOnlineSARSALambda(mc, M=M, tdr=tdr, alpha=alpha, approach='epsilon-greedy')
    Gs = np.zeros(alg.num_episodes)

    for _ in range(num_trials):
        alg.reset()
        Gs += np.array(alg.run())

    Gs /= num_trials

    plt.clf()
    plt.plot(range(len(Gs)), Gs)
    plt.savefig(f'experiments/SARSALambda/MountainCar/M-{M}-lambda-{tdr}-alpha-{alpha}.png', facecolor='white')

hp_settings = [
    (2, 8, 0.00001, 0.01),
    (3, 8, 0.00001, 0.01),
    (2, 6, 0.00001, 0.01),
    (3, 8, 0.001, 0.01),
    (3, 8, 0.001, 0.1),
    (3, 8, 0.001, 0.05),
]
for h in hp_settings:
    M, n, alpha, epsilon = h
    alg = ESGNStepSARSA(mc, M=M, n=n, alpha=alpha, epsilon=epsilon, approach='epsilon-greedy')
    Gs = np.zeros(alg.num_episodes)

    for _ in range(num_trials):
        alg.reset()
        Gs += np.array(alg.run())

    Gs /= num_trials

    plt.clf()
    plt.plot(range(len(Gs)), Gs)
    plt.savefig(f'experiments/SARSALambda/MountainCar/M-{M}-n-{n}-alpha-{alpha}-epsilon-{epsilon}.png', facecolor='white')