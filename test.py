import numba
import random
import numpy as np

@numba.jit
def mcmove_cpu(L, mcstep, beta=10):
    spin = np.empty((L, L), dtype=np.int16)
    for idx in np.ndindex((L, L)): 
        spin[idx] = 2 * np.random.randint(2) - 1

    for _ in range(L ** 2):
        x, y = random.randint(0, L - 1), random.randint(0, L - 1)
        s = spin[x, y]

        xpp = (x + 1) if (x + 1) < L else 0
        ypp = (y + 1) if (y + 1) < L else 0
        xnn = (x - 1) if (x - 1) >= 0 else (L - 1)
        ynn = (y - 1) if (y - 1) >= 0 else (L - 1)
        R = spin[xpp, y] + spin[x, ypp] + spin[xnn, y] + spin[x, ynn]

        dH = 2 * s * R
        if dH < 0:
            s = -s
        elif random.random() < np.exp(-beta * dH):
            s = -s
        spin[x, y] = s
    return spin

@numba.jit
def cpu(spin, L, mcstep):
    energy = 0
    for x in range(L):
        for y in range(L):
            S = spin[x, y]
            xpp = (x + 1) if (x + 1) < L else 0
            ypp = (y + 1) if (y + 1) < L else 0
            xnn = (x - 1) if (x - 1) >= 0 else (L - 1)
            ynn = (y - 1) if (y - 1) >= 0 else (L - 1)

            R = spin[xpp, y] + spin[x, ypp] + spin[xnn, y] + spin[x, ynn]
            energy -= R * S
    np_cpu(spin, L, mcstep)
    return energy / 4


def np_cpu(spin, L, mcstep) -> np.array:
    R = np.roll(spin, 1, 0) + np.roll(spin, -1, 0) + np.roll(spin, 1, 1) + np.roll(spin, -1, 1)
    return np.sum(-R * spin) / 4
