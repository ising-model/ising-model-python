import numba
import random
import numpy as np


@numba.jit
def _metropolis(spin, L, beta):
    for _ in range(L ** 2):
        # randomly sample a spin
        x, y = random.randint(0, L - 1), random.randint(0, L - 1)
        s = spin[x, y]

        # Sum of the spins of nearest neighbors
        xpp = (x + 1) if (x + 1) < L else 0
        ypp = (y + 1) if (y + 1) < L else 0
        xnn = (x - 1) if (x - 1) >= 0 else (L - 1)
        ynn = (y - 1) if (y - 1) >= 0 else (L - 1)
        R = spin[xpp, y] + spin[x, ypp] + spin[xnn, y] + spin[x, ynn]

        # Check Metropolis-Hastings algorithm for more details
        dH = 2 * s * R  # Change of the Hamiltionian after flippling the selected spin
        if dH < 0:      # Probability of the flipped state is higher -> flip the spin
            s = -s
        elif np.random.rand() < np.exp(-beta * dH): # Flip randomly according to the temperature
            s = -s
        spin[x, y] = s


class MonteCarlo2D:
    def __init__(self, args):
        self.L = args.size
        self.eqstep = args.eqstep
        self.mcstep = args.mcstep

        self.n1 = 1.0 / (self.mcstep * self.L ** 2)
        self.n2 = 1.0 / (self.mcstep ** 2 * self.L ** 2) 
        self.spin = None

    def _init_spin(self):
        return 2 * np.random.randint(2, size=(self.L, self.L)) - 1
    
    # Calculate energy using neighbors
    def _calc_energy(self, spin):
        R = np.roll(spin, 1, axis=0) + np.roll(spin, -1, axis=0) + np.roll(spin, 1, axis=1) + np.roll(spin, -1, axis=1)
        return np.sum(-R * spin) / 4
    
    def _calc_magnetization(self, spin):
        return np.sum(spin)
    
    def simulate(self, beta):
        E1, M1, E2, M2 = 0, 0, 0, 0
        # Initialize the lattice randomly
        spin = self._init_spin()
        # Equilibration steps
        for _ in range(self.eqstep):
            _metropolis(spin, self.L, beta)
        # Monte Carlo steps
        for _ in range(self.mcstep):
            _metropolis(spin, self.L, beta)
            E = self._calc_energy(spin)
            M = self._calc_magnetization(spin)
            E1 += E
            M1 += M
            E2 += E ** 2
            M2 += M ** 2

        return self.n1 * E1, self.n1 * M1, (self.n1 * E2 - self.n2 * E1 * E1) * beta ** 2, (self.n1 * M2 - self.n2 * M1 * M1) * beta