# Python Implementation of Ising model in 2D and 3D

[![CodeFactor](https://www.codefactor.io/repository/github/ising-model/ising-model-3d/badge)](https://www.codefactor.io/repository/github/ising-model/ising-model-3d)

Python code implementing Markov Chain Monte Carlo for 2D and 3D square-lattice Ising model.

- [x] Numba JIT compiling supported
- [x] Multiprocessing supported

> [!Warning]
> Experiments for a large scale 3D-lattice Ising model consume *a lot of* energy and time. If you do not have a desktop, we strongly recommend you to steal a server from a compiler lab or a deep learning lab.

## Result

![Result of 3D lattice](./result/plot_L30_D3_EQ16000_MC16000.png)

It is possible to calculate mean energy, magnetization, specific heat, and susceptibility at various temperatures and save it to a csv file and a plot.

We ran this code for 16000 equilibration steps and 16000 Monte Carlo steps on a 30 x 30 x 30 lattice to get the result above.

## Install requirements

To install requirements, run the command below:

```bash
pip install -r requirements.txt
```

## Options

### Monte Carlo arguments
- size (default=30): Length of the lattice -> L
- dim (default=3): Dimension of the lattice -> D
- init_temp (default=1.5): Initial temperature -> T_0
- final_temp (default=6.5): Final temperature -> T_f
- temp_step (default=0.04): Temperature step -> dT
- eqstep (default=1000): Number of equilibration steps
- mcstep (default=1000): Number of Monte Carlo steps

### Misc
- seed (default=0): Random seed
- n_proc (default=0): Number of processes for multiprocessing. If n_proc = 0, then use all CPU cores available
- no_record (default=record): Whether to record the result or not
- no_plot (default=plot): Whether to plot the result or not

## Examples

To run experiments, run the command below:

### 2D-lattice Ising model

```bash
python main.py --size 30 --dim 2 --init_temp 1.5 --final_temp 3.5 --temp_step 0.02 --eqstep 1000 --mcstep 1000
```

### 3D-lattice Ising model
```bash
python main.py --size 30 --dim 3 --init_temp 1.5 --final_temp 6.5 --temp_step 0.04 --eqstep 3000 --mcstep 3000
```

## Future works to be done
We want to parallelize the sampling procedure using GPU.

We also want to speed up the process using various techniques (e.g. importance sampling). 

If you have abundant knowledge of those techniques, please contact us!