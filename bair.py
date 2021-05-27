import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
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

def run_simulation(user_model, delta, alpha=1.0, m=1, verbose=True):
    K = len(user_model.mu)
    N_1 = (2*(K-1)/delta)**(1/ alpha) /rho
    # N_1= 0
    if verbose:
        print('Phase-1 steps:', N_1)
    #### phase 1 ####
    # initialization
    R = 1
    N = 0  # total number of acceptance
    F = [i for i in range(K)]
    while N < N_1:
        if verbose:
            print('iteration ', N)
        if not F:
            F = [i for i in range(K)]
            R += 1
        for i in F:
            if verbose:
                print('Arm tried:', i)
                print('user_model.global_time:', user_model.global_time)
                print('user_model.acceptance:', user_model.num_acceptance_per_arm)
                print('user_model.rejection:', user_model.num_rejection_per_arm)
                print('user_model.ucb:', user_model.ucb)
                print('user_model.mean:', user_model.empirical_mean_per_arm)
                print('user_model.lcb:', user_model.lcb)

            if R <= 1:
                accepted = user_model.isUserAccept(i)
                if not accepted:
                    F.remove(i)
                else:
                    N += 1
            else:
                while True:
                    accepted = user_model.isUserAccept(i)
                    if not accepted:
                        F.remove(i)
                        break
                    else:
                        N += 1
                        if N >= N_1:
                            break
    if verbose:
        print('Phase-2 starts:')
    #### phase 2 ####
    # initializationF
    # F = [i for i in range(K)]
    # while len(F) > 1:
    #     if verbose:
    #         print(F)
    #     mask = np.ones(K) * 1e8
    #     mask[F] = 0
    #     mask[F] += user_model.num_acceptance_per_arm[F]
    #     i = mask.argmin()
    #     accepted = user_model.isUserAccept(i)
    #     if not accepted:
    #         F.remove(i)
    # last_arm = F.pop()

    F = dict(zip(range(K), [m] * K))
    while len(F) > 1:
        if verbose:
            print(F)
        for i in range(K):
            if i in F.keys():
                accepted = user_model.isUserAccept(i)
                if not accepted:
                    F[i] -= 1
                    if F[i] < 0:
                        F.pop(i)
                        if len(F) == 1:
                            break;

    last_arm = list(F.keys())[0]

    best_arm = np.argmax(user_model.mu)
    print("best arm is {}, algorithm finds {} in {} rounds".format(best_arm, last_arm, user_model.global_time))
    return last_arm == best_arm, user_model.global_time, user_model.num_acceptance_total

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
        best_arm = np.argmax(user_model.mu)
        last_arm = pd.value_counts(self.actions).idxmax()
        print("best arm is {}, algorithm finds {} in {} rounds".format(best_arm, last_arm, user_model.global_time))
        return last_arm == best_arm, user_model.global_time, user_model.num_acceptance_total

def generate_mu(K, min_gap):
    p = []
    for key in range(K):
        r = np.random.rand()
        p.append(r)
    gap = sorted(p)[-1]-sorted(p)[-2]
    p[np.argmax(p)] = p[np.argmax(p)] - gap + min_gap
    return p

if __name__ == '__main__':

    # environment settings
    parser = argparse.ArgumentParser(description="BAIR test")
    parser.add_argument('--rho', default=1)
    parser.add_argument('--alpha', default=1)
    parser.add_argument('--dp', default=1)
    parser.add_argument('--m', default=1)
    parser.add_argument('--min_gap', default=0.5)
    parser.add_argument('--num_trials', default=1000)
    parser.add_argument('--T', default=1000)
    parser.add_argument('--alg', default='bair')
    parser.add_argument('--delta', default=0.1)
    parser.add_argument('--K', default=2)
    parser.add_argument('--N1_amp', default=1)
    parser.add_argument('--max_ite', default=100000)

    args = vars(parser.parse_args())

    rho = float(args['rho'])
    alpha = float(args['alpha'])
    detpro = float(args['dp'])
    m = float(args['m'])
    min_gap = float(args['min_gap'])
    num_trials = int(args['num_trials'])
    TT = int(args['T'])
    alg = args['alg']
    delta = float(args['delta'])
    K = int(args['K'])
    N1_amp = float(args['N1_amp'])
    max_ite = int(args['max_ite'])

    m_in_phase2 = m*8*detpro*(1-detpro)/((2*detpro-1)**2)*np.log(2*K/delta)

    res_ls = []
    T = []
    N = []
    print("alg, min_gap, delta, K", alg, min_gap, delta, K)
    for run in tqdm(range(num_trials)):
        mu = generate_mu(K=K, min_gap=min_gap)
        user_model = User(mu=mu, rho=rho, alpha=alpha, determ_pro=detpro)
        user_model.initialize()
        if alg == 'tas':
            res, t, n = run_track_and_stop(user_model, delta, MAX_ITE=max_ite)
        elif alg == 'bair':
            res, t, n = run_simulation(user_model, delta, alpha=alpha, m=m_in_phase2, verbose=False)
        elif alg == 'uni':
            res, t, n = uniform_explore(user_model, TT)
        elif alg == 'exp3':
            exp3 = EXP3(K=K, user_model=user_model)
            res, t, n = exp3.run(eps=np.sqrt(K*np.log(K)/TT), T=TT, verbose=False)
        res_ls.append(res)
        if t >= 0 and n >= 0:
            T.append(t)
            N.append(t - n)
        print(np.mean(T), np.mean(N) / np.mean(T) * 100, np.mean(res_ls))

# python-jl bair.py --alg=bair --min_gap=0.5 --delta=0.1 --K=20 --alpha=0.1