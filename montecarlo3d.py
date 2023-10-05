import numba
import random
import numpy as np


@numba.jit(nopython=True)
def _metropolis3d(spin, L, beta):
    for _ in range(L ** 3):
        # randomly sample a spin
        x, y, z = random.randint(0, L - 1), random.randint(0, L - 1), random.randint(0, L - 1)
        s = spin[x, y, z]

        # Sum of the spins of nearest neighbors
        xpp = (x + 1) if (x + 1) < L else 0
        ypp = (y + 1) if (y + 1) < L else 0
        zpp = (z + 1) if (z + 1) < L else 0
        xnn = (x - 1) if (x - 1) >= 0 else (L - 1)
        ynn = (y - 1) if (y - 1) >= 0 else (L - 1)
        znn = (z - 1) if (z - 1) >= 0 else (L - 1)
        R = spin[xpp, y, z] + spin[x, ypp, z] + spin[x, y, zpp] \
            + spin[xnn, y, z] + spin[x, ynn, z] + spin[x, y, znn]

        # Check Metropolis-Hastings algorithm for more details
        dH = 2 * s * R  # Change of the Hamiltionian after flippling the selected spin
        if dH < 0:      # Probability of the flipped state is higher -> flip the spin
            s = -s
        elif random.random() < np.exp(-beta * dH):  # Flip randomly according to the temperature
            s = -s
        spin[x, y, z] = s


class MonteCarlo3D:
    def __init__(self, args):
        self.L = args.size
        self.eqstep = args.eqstep
        self.mcstep = args.mcstep

        self.n1 = 1.0 / (self.mcstep * self.L ** 3)
        self.n2 = 1.0 / (self.mcstep ** 2 * self.L ** 3) 

    def _init_spin(self):
        return 2 * np.random.randint(2, size=(self.L, self.L, self.L)) - 1
    
    # Calculate energy using neighbors
    def _calc_energy(self, spin):
        R = np.roll(spin, 1, axis=0) + np.roll(spin, -1, axis=0) \
            + np.roll(spin, 1, axis=1) + np.roll(spin, -1, axis=1) \
            + np.roll(spin, 1, axis=2) + np.roll(spin, -1, axis=2)
        return np.sum(-R * spin) / 6
    
    def _calc_magnetization(self, spin):
        return np.sum(spin)
    
    # Monte Carlo CPU version
    def simulate(self, beta):
        E1, M1, E2, M2 = 0, 0, 0, 0
        # Initialize the lattice randomly
        spin = self._init_spin()
        # Equilibration steps
        for _ in range(self.eqstep):
            _metropolis3d(spin, self.L, beta)
        # Monte Carlo steps
        for _ in range(self.mcstep):
            _metropolis3d(spin, self.L, beta)
            E = self._calc_energy(spin)
            M = self._calc_magnetization(spin)
            E1 += E
            M1 += M
            E2 += self.n1 * E ** 2
            M2 += self.n1 * M ** 2

        return self.n1 * E1, self.n1 * M1, (E2 - self.n2 * E1 * E1) * beta ** 2, (M2 - self.n2 * M1 * M1) * beta