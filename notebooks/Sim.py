import jax.numpy as jnp
from jax import random, jit, grad
from jax.config import config

from time import time
import csv

config.update("jax_enable_x64", True)

from optispd.manifold import SPD
from optispd.minimizer import minimizer
from utils import *

rng = random.PRNGKey(42)

filename = "results_{}_{}.csv"
columns = ['p',  'n', 'par', 'variance', 'variance_rot', 'time_kar', 'time_rot']

hype = {
    'dist': ['wishart',
             #'invwishart',
             'riem-gauss'],
    'p': [5], #10, 20, 50],
    'n': [50, 100, 500]
}

sig = jnp.array([0.1, 0.5])
dof = jnp.array([2, 20])

nrep = 50

total_shape = tuple([len(i) for k, i in hype.items()]) + (len(sig), ) + (nrep, )
total_counts = jnp.prod(jnp.array(total_shape))
print(total_shape, '->', total_counts)

rng, *key = random.split(rng, total_counts + 1)

compute_cost = lambda x, y: jnp.sum(dist(x, y)**2)

for distr in hype['dist']:
    for p in hype['p']:
        if (distr == 'wishart') or (distr == 'invwishart'):
            pars = jnp.array(dof * p + 1, dtype=int)
        elif distr == 'riem-gauss':
            pars = sig
        else:
            raise ValueError(f'Unrecognized dist = {distr}!')
        df_content = []
        
        man = SPD(p)
        opti = minimizer(man, 'rcg', verbosity=0, maxiter=50)

        for n in hype['n']:

            for par in pars:
                print(f"Distribution: {distr} with parameter {par}  |  "
                      f"Matrix dimension: {p}  |  "
                      f"Number of samples: {n}")
                karc_dist = 0
                progr = 0
                karc_time, rota_time, samp_time = 0, 0, 0

                if distr == 'wishart':
                    gen_samp = jit(lambda k, cent: Wishart(k, dof=par, scale=cent / par, shape=(n,)))
                elif distr == 'invwishart':
                    gen_samp = jit(lambda k, cent: InvWishart(k, dof=par, scale=cent * (par - p - 1), shape=(n,)))
                elif distr == 'riem-gauss':
                    gen_samp = jit(lambda k, cent: generate_sample(k, cent, p, n, par, thinning=10, chains=1))
                else:
                    raise ValueError(f'Unrecognized dist = {distr}!')

                for r in range(nrep):
                    k1, k2 = random.split(key[progr])
                    center = generate_center(k1, p, maxexp=5)

                    tic = time()
                    sample = gen_samp(k2, center)
                    samp_time += time() - tic

                    tic = time()
                    fun = lambda x: compute_cost(x, sample)
                    init = jnp.mean(sample, axis=0)
                    res = opti.solve(fun, x=init)
                    karcher = res.x
                    karc_t = time() - tic
                    karc_time += karc_t
                    karc_dist += dist(karcher, center)

                    tic = time()
                    estimat = optimal_reference(sample, return_eigs=False)
                    rsample = optimal_rotation(sample, estimat)
                    rota_t = time() - tic
                    rota_time += rota_t

                    d_ori = compute_cost(center, sample) / (n - 1)
                    d_rot = compute_cost(estimat, rsample) / (n - 1)

                    df_content.append([p, distr, n, par, float(d_ori), float(d_rot), float(karc_t), float(rota_t)])

                    progr += 1
                    print("Progress <{:.0f}%> [{}] \r".format(100 * progr / nrep, 'o' * progr + '.' * (nrep - progr)), end="")
                print(f"\nAverage distance of Karcher barycenter {karc_dist / nrep:.2e}")
                print(f"Times:\n\tSample {samp_time / nrep:.3f} s,\n\t"
                      f"Compute karcher {karc_time / nrep:.3f} s,\n\t"
                      f"Compute rotation {rota_time / nrep:.3f} s,\n\t"
                     )


        with open(filename.format(distr, p), "w", newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(columns)
            for line in df_content:
                writer.writerow(line)
