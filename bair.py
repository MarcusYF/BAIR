import numpy as np
from tqdm import tqdm
import argparse
from algorithms import *
from User import User


def run_simulation(user_model, delta, alpha=1.0, m=1, N1=None, verbose=True):
    K = len(user_model.mu)
    if N1 is None:
        N_1 = (2*(K-1)/delta)**(1/ alpha) /rho
    else:
        N_1 = N1
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
                            break

    last_arm = list(F.keys())[0]

    best_arm = np.argmax(user_model.mu)
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
    parser.add_argument('--N1', default=1)
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
    N1 = float(args['N1'])
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
            res, t, n = run_simulation(user_model, delta, alpha=alpha, m=m_in_phase2, N1=N1, verbose=False)
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

