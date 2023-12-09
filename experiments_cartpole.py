from MDP import CartPole
from RLAlgorithms import ESGNStepSARSA, TrueOnlineSARSALambda
import matplotlib.pyplot as plt
import numpy as np

num_trials = 20
cp = CartPole()


for approach in ['epsilon-greedy', 'softmax']:
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 0.99]:
        print('lambda:',t)
        alg = TrueOnlineSARSALambda(cp, M=10, tdr=t, approach=approach)
        Gs = np.zeros(alg.num_episodes)

        for _ in range(num_trials):
            alg.reset()
            Gs += np.array(alg.run())

        Gs /= num_trials

        plt.clf()
        plt.plot(range(len(Gs)), Gs)
        plt.savefig(f'experiments/SARSALambda-CartPole-Lambda-{t}-{approach}.png', facecolor='white')


for alpha in [0.2, 0.1, 0.05]:
    print('alpha:',alpha)
    alg = TrueOnlineSARSALambda(cp, M=10, tdr=0.6, alpha=alpha)
    Gs = np.zeros(alg.num_episodes)

    for _ in range(num_trials):
        alg.reset()
        Gs += np.array(alg.run())

    Gs /= num_trials

    plt.clf()
    plt.plot(range(len(Gs)), Gs)
    plt.savefig(f'experiments/SARSALambda/CartPole/alpha-{alpha}.png', facecolor='white')

for M in [2, 3, 5, 8, 10, 15]:
    print('M:',M)
    alg = ESGNStepSARSA(cp, M=M, n=5, alpha=0.01, epsilon=0.1, approach='epsilon-greedy')
    Gs = np.zeros(alg.num_episodes)

    for _ in range(num_trials):
        alg.reset()
        Gs += np.array(alg.run())

    Gs /= num_trials

    plt.clf()
    plt.plot(range(len(Gs)), Gs)
    plt.savefig(f'experiments/nStepSARSA/CartPole/M-{M}.png', facecolor='white')

for epsilon in [0.2, 0.1, 0.05, 0.01, 0.005, 0.001]:
    print('epsilon:',epsilon)
    alg = ESGNStepSARSA(cp, M=10, n=5, alpha=0.01, approach='softmax')
    Gs = np.zeros(alg.num_episodes)

    for _ in range(num_trials):
        alg.reset()
        Gs += np.array(alg.run())

    Gs /= num_trials

    plt.clf()
    plt.plot(range(len(Gs)), Gs)
    plt.savefig(f'experiments/nStepSARSA/CartPole/epsilon-{epsilon}.png', facecolor='white')

for approach in ['epsilon-greedy', 'softmax'][1:]:
    for alpha in [0.1, 0.01, 0.001]:
        print('alpha:', alpha)
        alg = ESGNStepSARSA(cp, M=10, n=5, alpha=alpha, epsilon=0.1, approach=approach)
        Gs = np.zeros(alg.num_episodes)

        for _ in range(num_trials):
            alg.reset()
            Gs += np.array(alg.run())

        Gs /= num_trials

        plt.clf()
        plt.plot(range(len(Gs)), Gs)
        plt.savefig(f'experiments/nStepSARSA/CartPole/alpha-{alpha}-{approach}.png', facecolor='white')