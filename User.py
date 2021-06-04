import numpy as np


class User:
    def __init__(self, mu, rho, alpha, determ_pro):
        self.rho0 = rho
        self.alpha = alpha
        self.determ_pro = determ_pro
        self.K = len(mu)  # num of arms
        self.mu = mu  # vector of the ground truth mean reward for each arm
        self.sigma = np.ones_like(mu)  # vector of standard deviation for each arm's reward distribution
        self.global_time = self.K  # total number of interactions
        self.num_acceptance_total = 0.0
        self.empirical_mean_per_arm = np.zeros(self.K)
        self.num_acceptance_per_arm = np.zeros(self.K)
        self.num_rejection_per_arm = np.zeros(self.K)
        self.accept_ratio = 0
        self.ucb = np.zeros(self.K)
        self.lcb = np.zeros(self.K)

    def initialize(self):
        # pull each arm once
        for i in range(self.K):
            sampled_reward = np.random.normal(self.mu[i], self.sigma[i])
            self.empirical_mean_per_arm[i] = sampled_reward
            self.num_acceptance_per_arm[i] = 1
            self.num_acceptance_total += 0
        self.ucb = self.empirical_mean_per_arm + np.sqrt(self.trust_func() / self.num_acceptance_per_arm)
        self.lcb = 2 * self.empirical_mean_per_arm - self.ucb
        self.accept_ratio = 1

    def rho(self):
        return self.rho0 + self.accept_ratio * 1

    def trust_func(self):
        """
        trust function / numerator in CB term
        :return:
        """
        return np.max([0, 2*self.alpha*np.log(  self.rho() * max( self.num_acceptance_total, 1) ) ]  )

    def isUserAccept(self, arm_id):
        """
        user decides whether accept the recommended arm or not
        by looking at whether there is another arm whose LCB is larger than UCB of the recommended arm
        :param arm_id:
        :return:
        """
        self.global_time += 1

        if self.determ_pro > np.random.rand():
            gamma = self.trust_func()
            CB = [np.sqrt(gamma / float(max(1, self.num_acceptance_per_arm[i]))) for i in range(self.K)]
            UCB_i = self.empirical_mean_per_arm[arm_id] + CB[arm_id]
            # find the arm with the highest lcb
            LCB_j = max([self.empirical_mean_per_arm[i] - CB[i] for i in range(self.K)])
            if LCB_j > UCB_i:
                return False
            # no other arm j has LCB larger than UCB of arm i
            self.num_acceptance_total += 1.0
            self.accept_ratio = (self.K + self.num_acceptance_total) / (self.K + self.global_time)
            # observe reward of arm i
            sampled_reward = np.random.normal(self.mu[arm_id], self.sigma[arm_id])
            self.empirical_mean_per_arm[arm_id] = (self.empirical_mean_per_arm[arm_id] * self.num_acceptance_per_arm[
                arm_id] + sampled_reward) / (self.num_acceptance_per_arm[arm_id] + 1.0)
            self.num_acceptance_per_arm[arm_id] += 1.0
            return True
        else:
            if np.random.rand() > 0.5:
                return False
            else:
                self.num_acceptance_total += 1.0
                self.accept_ratio = (self.K + self.num_acceptance_total) / (self.K + self.global_time)
                # observe reward of arm i
                sampled_reward = np.random.normal(self.mu[arm_id], self.sigma[arm_id])
                self.empirical_mean_per_arm[arm_id] = (self.empirical_mean_per_arm[arm_id] *
                                                       self.num_acceptance_per_arm[
                                                           arm_id] + sampled_reward) / (
                                                                  self.num_acceptance_per_arm[arm_id] + 1.0)
                self.num_acceptance_per_arm[arm_id] += 1.0
                return True

    def isUserAccept2(self, arm_id):
        """
        user decides whether accept the recommended arm or not
        by looking at whether there is another arm whose LCB is larger than UCB of the recommended arm
        :param arm_id:
        :return:
        """

        self.global_time += 1
        CB_i = np.sqrt(self.trust_func() / float(max(1, self.num_acceptance_per_arm[arm_id])))
        UCB_i = self.empirical_mean_per_arm[arm_id] + CB_i
        for j in range(self.K):
            if j != arm_id:
                CB_j = np.sqrt(self.trust_func() / float(max(1, self.num_acceptance_per_arm[j])))
                LCB_j = self.empirical_mean_per_arm[j] - CB_j
                if LCB_j > UCB_i:
                    return False
        # no other arm j has LCB larger than UCB of arm i
        self.num_acceptance_total += 1.0
        self.accept_ratio = (self.K + self.num_acceptance_total) / (self.K + self.global_time)
        # observe reward of arm i
        sampled_reward = np.random.normal(self.mu[arm_id], self.sigma[arm_id])
        self.empirical_mean_per_arm[arm_id] = (self.empirical_mean_per_arm[arm_id] * self.num_acceptance_per_arm[arm_id] + sampled_reward) / (self.num_acceptance_per_arm[arm_id] + 1.0)
        self.num_acceptance_per_arm[arm_id] += 1.0
        return True