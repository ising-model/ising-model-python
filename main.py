import os
import csv
import time
import argparse
import random
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
from contextlib import contextmanager

from montecarlo2d import MonteCarlo2D
from montecarlo3d import MonteCarlo3D


def args_parser():
    parser = argparse.ArgumentParser()
    # monte carlo sampling arguments
    parser.add_argument('--size', type=int, default=30, help="Length of the lattice: L")
    parser.add_argument('--dim', type=int, default=3, help="Dimension of the lattice: D")
    parser.add_argument('--init_temp', type=float, default=1.5, help="Initial temperature: T_0")
    parser.add_argument('--final_temp', type=float, default=6.5, help="Final temperature: T_f")
    parser.add_argument('--temp_step', type=float, default=0.04, help="Temperature step: dT")
    parser.add_argument('--eqstep', type=int, default=1000, help="Number of equilibration steps")
    parser.add_argument('--mcstep', type=int, default=1000, help="Number of Monte Carlo steps")

    # misc
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument('--n_proc', type=int, default=0, help="Number of processors for multiprocessing")
    parser.add_argument('--no_record', action='store_true', help='Whether to record or not (default: record)')
    parser.add_argument('--no_plot', action='store_true', help='Whether to plot the result or not (default: no plotting)')

    args = parser.parse_args()
    return args


@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


# Monte Carlo task
def mcmc_task(args, Ts):
    Es, Ms, Cs, Xs = [], [], [], []
    if args.dim == 2:
        m = MonteCarlo2D(args)
    else:
        m = MonteCarlo3D(args)
    pbar = tqdm(desc="Progress: ".format(id), total=len(Ts))
    for T in Ts:
        E, M, C, X = m.simulate(1 / T)
        Es.append(E)
        Ms.append(abs(M))
        Cs.append(C)
        Xs.append(X)
        pbar.update(1)
    return Es, Ms, Cs, Xs


if __name__ == "__main__":
    # parse args and set seed
    args = args_parser()
    print("> Settings: ", args)
    assert args.dim < 4 and args.dim > 1, "1 < Dimension of the lattice < 4"
    n_proc = mp.cpu_count() if args.n_proc == 0 else args.n_proc
    print("> Number of processes: ", n_proc)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_grad_enabled(False)
    
    # Temperature settings
    T_0 = args.init_temp
    T_f = args.final_temp
    dT = args.temp_step

    # Monte Carlo in a pool
    start = time.time()
    args_list = [args for _ in range(n_proc)]
    NT = int((T_f - T_0) / dT) + 1
    Ts = [T_0 + dT * step for step in range(NT)]
    Ts_list = np.array_split(np.array(Ts), n_proc)
    with poolcontext(processes=n_proc) as pool:
        Es_list, Ms_list, Cs_list, Xs_list = zip(*pool.starmap(mcmc_task, zip(args_list, Ts_list)))
    Es, Ms, Cs, Xs = sum(Es_list, []), sum(Ms_list, []), sum(Cs_list, []), sum(Xs_list, [])
    print("\n> Elapsed time: {:4f}s".format(time.time() - start))

    # Record and plot the result
    rootpath = './result'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)

    # Save the result into csv file
    if not args.no_record:
        csv_name = rootpath + "/result_L{}_D{}_EQ{}_MC{}.csv".format(args.size, args.dim, args.eqstep, args.mcstep)
        f = open(csv_name, 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['T', 'E', 'M', 'C', 'X'])
        for t, e, m, c, x in zip(Ts, Es, Ms, Cs, Xs):
            writer.writerow([t, e, m, c, x])
        f.close()
        print("Saved the result into {}.".format(csv_name))

    # Save the result into a plot
    if not args.no_plot:
        fig = plt.figure(figsize=(18, 10))
        sp = fig.add_subplot(2, 2, 1)
        plt.scatter(Ts, Es, s=50, marker='o', color='IndianRed')
        plt.xlabel("Temperature (T)", fontsize=20)
        plt.ylabel("Energy", fontsize=20)
        plt.axis('tight')

        sp = fig.add_subplot(2, 2, 2)
        plt.scatter(Ts, Ms, s=50, marker='o', color='RoyalBlue')
        plt.xlabel("Temperature (T)", fontsize=20)
        plt.ylabel("Magnetization", fontsize=20)
        plt.axis('tight')

        sp = fig.add_subplot(2, 2, 3)
        plt.scatter(Ts, Cs, s=50, marker='o', color='IndianRed')
        plt.xlabel("Temperature (T)", fontsize=20)
        plt.ylabel("Specific Heat", fontsize=20)
        plt.axis('tight')

        sp = fig.add_subplot(2, 2, 4)
        plt.scatter(Ts, Xs, s=50, marker='o', color='RoyalBlue')
        plt.xlabel("Temperature (T)", fontsize=20)
        plt.ylabel("Susceptibility", fontsize=20)
        plt.axis('tight')

        plot_name = rootpath + "/plot_L{}_D{}_EQ{}_MC{}.png".format(args.size, args.dim, args.eqstep, args.mcstep)
        plt.savefig(plot_name)
        plt.clf()
        print("Saved the plot into {}.".format(plot_name))