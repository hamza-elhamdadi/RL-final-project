from MDP import MountainCar, CartPole
from RLAlgorithms import ESGNStepSARSA, TrueOnlineSARSALambda
# import matplotlib.pyplot as plt
import numpy as np

cp = CartPole()

# alg_q = TrueOnlineSARSALambda(cp, M=10, tdr=0.6)
# alg_q.run()

alg = ESGNStepSARSA(cp, M=10, n=5, alpha=0.001, approach='epsilon-greedy')
# alg.w = alg_q.w
# alg.epsilon = alg_q.epsilon
Gs = np.array(alg.run())

plt.clf()
plt.plot(range(len(Gs)), Gs)