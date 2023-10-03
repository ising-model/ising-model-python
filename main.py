import os
import csv
import argparse
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from montecarlo2d import MonteCarlo2D
from montecarlo3d import MonteCarlo3D


def args_parser():
    parser = argparse.ArgumentParser()
    # monte carlo sampling arguments
    parser.add_argument('--size', type=int, default=30, help="Length of the lattice: L")
    parser.add_argument('--parallel_spins', type=int, default=10, help="Number of spins to flip simultaneously")
    parser.add_argument('--dim', type=int, default=3, help="Dimension of the lattice: D")
    parser.add_argument('--init_temp', type=float, default=1.5, help="Initial temperature: T_0")
    parser.add_argument('--final_temp', type=float, default=6.5, help="Final temperature: T_f")
    parser.add_argument('--temp_step', type=float, default=0.04, help="Temperature step: dT")
    parser.add_argument('--eqstep', type=int, default=1000, help="Number of equilibration steps")
    parser.add_argument('--mcstep', type=int, default=1000, help="Number of Monte Carlo steps")

    # misc
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--gpu', action='store_true', help='whether to use gpu or not (default: cpu)')
    parser.add_argument('--no_record', action='store_true', help='whether to record or not (default: record)')
    parser.add_argument('--no_plot', action='store_true', help='whether to plot the result or not (default: no plotting)')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # parse args and set seed
    args = args_parser()
    print("> Settings: ", args)
    assert args.dim < 4 and args.dim > 1, "1 < Dimension of the lattice < 4"
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Temperature settings
    T = args.init_temp
    T_f = args.final_temp
    dT = args.temp_step

    # Monte Carlo sampling
    pbar = tqdm(total=(T_f - T) // dT + 1)
    Ts, Es, Ms, Cs, Xs = [], [], [], [], []
    if args.dim == 2:
        m = MonteCarlo2D(args)
    else:
        m = MonteCarlo3D(args)
    while T <= T_f:
        E, M, C, X = m.simulate(1 / T)
        Ts.append(T)
        Es.append(E)
        Ms.append(abs(M))
        Cs.append(C)
        Xs.append(X)
        T += dT
        pbar.update(1)

    # Record and plot the result
    rootpath = './result'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)

    # Save the result into csv file
    if not args.no_record:
        f = open(rootpath + "/result_L{}_D{}_EQ{}_MC{}.csv".format(args.size, args.dim, args.eqstep, args.mcstep), 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['T', 'E', 'M', 'C', 'X'])
        for t, e, m, c, x in zip(Ts, Es, Ms, Cs, Xs):
            writer.writerow([t, e, m, c, x])
        f.close()

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

        plt.savefig(rootpath + "/plot_L{}_D{}_EQ{}_MC{}.png".format(args.size, args.dim, args.eqstep, args.mcstep))
        plt.clf()