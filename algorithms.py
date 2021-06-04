import numpy as np
import pandas as pd
import julia

j = julia.Julia()
j.include("optimal_weights.jl")
OptimalWeights = j.eval("OptimalWeights")


def dBernoulli(p, q):
    res = 0
    if p != q:
        if p <= 0:
            p = np.finfo(float).eps
        if p >= 1:
            p = 1-np.finfo(float).eps
        res = p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))
    return res


def uniform_explore(user_model, T):
    K = len(user_model.mu)
    N = 0  # total number of acceptances
    R = 0  # total number of rejections
    F = np.zeros(K)
    while N + R < T:
        for i in range(K):
            accepted = user_model.isUserAccept(i)
            if not accepted:
                R += 1
            else:
                N += 1
                F[i] += 1
    #### phase 2 ####
    last_arm = np.argmax(F)
    best_arm = np.argmax(user_model.mu)
    print("best arm is {}, algorithm finds {} in {} rounds with {} rejections".format(best_arm, last_arm, user_model.global_time, R))
    return last_arm == best_arm, user_model.global_time, N


class EXP3:
    def __init__(self, K, user_model):
        self.K = K                          # number of actions
        self.user_model = user_model        # u = -l, the negation of the adversarial loss
        self.score = np.zeros((K, 1))       # score vector
        self.t = 0                          # global timer
        self.uniform = np.ones((K, 1)) / K  # score vector
        self.lr = []                        # learning rate
        self.eps = []                       # exploration probability
        self.option = ''                    # mode for different algorithms
        self.lrr = []                       # reverse learning rate
        self.actions = []
        self.accum_u = 0
        self.regret = 0

    def compute_prob(self):
        s = self.score - np.max(self.score, axis=0)
        p = np.exp(s) / sum(np.exp(s))
        return p * (1-self.eps[self.t]) + self.uniform * self.eps[self.t]  # (K, 1)

    def update(self, verbose=False):
        """
        execute one step of EXP3
        """
        # sample actions
        p = self.compute_prob()
        # print('Action Distr. \n{}:'.format(p.transpose()))
        action = np.random.multinomial(1, p[:, 0], size=1)[0].argmax()
        self.actions.append(action)
        # print('Sampled actions: \n{}, {}:'.format(action_1+1, action_2+1))
        # observe feedbacks
        u_ = self.user_model.isUserAccept(action) + 0.0
        self.accum_u += u_
        # update scores
        delta = self.lr[self.t] * u_ / p[action, 0]
        self.score[action, 0] += delta
        # print('Observed rewards: \n{}, {}:'.format(delta[0], delta[1]))
        if verbose and self.t % 100 == 0:
            print('Scores. \n{}:'.format(self.score[:, 0]))
        self.t += 1

    def run(self, eps=0.1, T=100, verbose=False):
        """
        run EXP3 for T steps
        """
        # Config learning rate
        lr = np.sqrt(2*np.log(self.K)/(self.K*T))
        self.lr = [lr] * T      # learning rate
        self.eps = [eps] * T    # exploration probability
        # Run algorithm
        for i in range(T):
            self.update(verbose=verbose)
            # print(i)
            # print('user mu:', user_model.mu)
            # print('user mean:', user_model.empirical_mean_per_arm)
            # print('user pull:', user_model.num_acceptance_per_arm)
            # print('score:', self.score)
        best_arm = np.argmax(self.user_model.mu)
        last_arm = pd.value_counts(self.actions).idxmax()
        print("best arm is {}, algorithm finds {} in {} rounds".format(best_arm, last_arm, self.user_model.global_time))
        return last_arm == best_arm, self.user_model.global_time, self.user_model.num_acceptance_total


def run_track_and_stop(user_model, delta, MAX_ITE):
    """
    :param mu: ground truth mean vector used to sample reward
    :param delta: confidence level
    :return:
    """
    mu = user_model.mu
    K = len(mu)  # number of arms
    num_pulls_per_arm = np.zeros(K)
    sum_reward_per_arm = np.zeros(K)
    # initialization
    for arm_id in range(K):
        num_pulls_per_arm[arm_id] = 1
        sum_reward_per_arm[arm_id] = 1
    t = K
    # condition = True
    arm_to_pull = 0
    while True:
        empirical_mean_per_arm = sum_reward_per_arm / num_pulls_per_arm
        best_arms = np.where(empirical_mean_per_arm == np.max(empirical_mean_per_arm))[0]
        # break tie by random if multiple best arms exist
        best_arm = np.random.choice(best_arms)
        if len(best_arms) > 1:
            arm_to_pull = best_arm
        else:
            # compute the stopping statistic
            num_pulls_best = num_pulls_per_arm[best_arm]
            sum_reward_best = sum_reward_per_arm[best_arm]
            empirical_mean_best = sum_reward_best / num_pulls_best
            MuMid = (sum_reward_best + sum_reward_per_arm) / (num_pulls_best + num_pulls_per_arm)
            Index = [arm_id for arm_id in range(K)]
            del Index[best_arm]
            Score = np.min([num_pulls_best*dBernoulli(empirical_mean_best, MuMid[i]) + num_pulls_per_arm[i]*dBernoulli(empirical_mean_per_arm[i],MuMid[i]) for i in Index])
            if Score > np.log((np.log(t)+1)/delta):
                print("stopping condition satisfied")
                # stop
                break
            elif t > MAX_ITE:
                # stop and output (0,0)
                # condition = False
                best_arm = 0
                print(num_pulls_per_arm)
                print(empirical_mean_per_arm)
                num_pulls_per_arm = np.zeros(K)
                return False, -1, -1
            else:
                if np.min(num_pulls_per_arm) <= np.max(np.sqrt(t)-K/2.0, 0):
                    # forced exploration
                    arm_to_pull = np.argmin(num_pulls_per_arm)
                else:
                    # continue and sample an arm
                    val, Dist = OptimalWeights(empirical_mean_per_arm, 1e-11)
                    # choice of the arm
                    arm_to_pull = np.argmax(Dist-num_pulls_per_arm/t)
        # draw the arm
        t += 1
        sum_reward_per_arm[arm_to_pull] += user_model.isUserAccept(arm_to_pull)
        num_pulls_per_arm[arm_to_pull] += 1
        # print("iter {}".format(t))
        # print("sum_reward_per_arm", sum_reward_per_arm)
        # print("num_pulls_per_arm ", num_pulls_per_arm)
        # print("empirical means   ", sum_reward_per_arm/num_pulls_per_arm)
    return best_arm == np.argmax(mu), t, sum(sum_reward_per_arm)