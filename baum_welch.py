import numpy as np
import pandas as pd

class BaumWelch():
    def __init__(self, observ_seq, p, q, b_0, max_iter = 100):
        self.observ_seq = observ_seq    # La séquence d'observation o = (o_1, o_2, ... , o_n)
        self.p = p                      # La probabilité de transition
        self.q = q                      # La probabilité d'émission
        self.b_0 = b_0                  # L'état initial 
        self.max_iter = max_iter        # Nombre maximal d'intération
    def forward(self):
        # Initialisation de  alpha_1(s) = q_(s,o_1) * b_0(s)
        alpha = np.zeros((self.p.shape[0], self.observ_seq.shape[0]))
        alpha[:,0] = self.b_0*self.q[:,self.observ_seq[0]]
        # calcul des  alpha_t(s) = q_(s,o_t) sum p(s', s) * alpha_(t-1) (s')
        for t in range(1, self.observ_seq.shape[0]):
            for s in range(self.q.shape[0]):
                alpha[s, t] = self.q[s, self.observ_seq[t]] * np.dot(self.p[:, s], alpha[:, t-1])
        return alpha
    def backward(self):
        # Initialisation de beta_T (s) = 1
        beta = np.zeros((self.observ_seq.shape[0], self.p.shape[0]))
        beta[self.observ_seq.shape[0] - 1] = np.ones((self.p.shape[0]))
        # calcul des  beta_t(s) = sum p(s, s') *  q(s', o_(t+1)) *beta_(t+1) (s')
        for t in range(self.observ_seq.shape[0] - 2, -1, -1):
            for s in range(self.q.shape[0]):
                beta[t, s] = np.dot(beta[t + 1] * self.q[:, self.observ_seq[t + 1]], self.p[s, :])
        return beta
    def fit(self):
        for _ in range(self.max_iter):
            alpha = self.forward()
            beta = self.backward()
            epsilon = np.zeros((self.p.shape[0], self.p.shape[0], len(self.observ_seq) - 1))
            for t in range(len(self.observ_seq) - 1):
                epsilon_1 = np.dot(np.dot(alpha[:, t].T, self.p) * self.q[:, self.observ_seq[t + 1]].T, beta[t + 1, :])
                for i in range(self.p.shape[0]):
                    epsilon_2 = alpha[i, t] * self.p[i, :] * self.q[:, self.observ_seq[t + 1]].T * beta[t + 1, :].T
                    epsilon[i, :, t] = epsilon_2 / epsilon_1
            gamma = np.sum(epsilon, axis=1)
            self.p = np.sum(epsilon, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
            gamma = np.hstack((gamma, np.sum(epsilon[:, :, len(self.observ_seq) - 2], axis=0).reshape((-1, 1))))
            for l in range(self.q.shape[1]):
                self.q[:, l] = np.sum(gamma[:, self.observ_seq == l], axis=1)
            self.q = np.divide(self.q, np.sum(gamma, axis=1).reshape((-1, 1)))
        return {"Transition":self.p, "Emission":self.q}