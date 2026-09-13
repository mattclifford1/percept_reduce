'''
the equal-optimisation-budget half of the data-efficiency experiment (TODO T1.2).

every committed run gives each cell 30 passes over *its own* training set, so the 1% cell gets
240 gradient steps against 22,500 for the 100% cell. every "less data" result is therefore also
a "~100x less optimisation" result, and the two cannot be told apart from one grid.

this runs the same grid with `epoch_scaling='equal_steps'`: epochs scale by 1/data_percent, so
every cell takes roughly the same number of gradient steps and only the *number of distinct
images* changes. **the difference between this grid and the fixed-budget one is the
data-efficiency result** -- neither grid answers it alone.

    python percept_loss/pipeline_CIFAR_BUDGET.py            # all three seeds
    python percept_loss/pipeline_CIFAR_BUDGET.py budget_s1  # one seed, for running in parallel

## Two things that make this run at all

`n_validations=16` -- a fixed number of evaluations per run rather than a fixed interval. at 1%
data equal_steps means 3000 epochs; at the usual validate_every=2 that is 1500 probes, roughly
six hours of sklearn for a cell whose training takes four minutes. probes are 95% of wall time
on this grid, so the eval cadence *is* the cost.

`props` passed explicitly -- the default split is derived from validate_every (FINDINGS B7), so
a computed cadence landing on 1 would silently swap the split to [0.89, 0.1, 0.01] and quietly
make these runs incomparable to the fixed-budget grid.

no collision with the fixed-budget grid: the scaled epoch count is in the run directory
(`_ep3000` vs `_ep30`), so the two budgets sit side by side.
'''
import sys

from percept_loss.networks import CIFAR_AUTOENCODERS
import percept_loss.pipeline.generic

SEEDS = [42, 1, 2]
PROPS = [0.4, 0.3, 0.3]      # identical to the fixed-budget grid -- do not let B7 pick this
N_VALIDATIONS = 16           # matches the 16 rows every fixed-budget run produces

budget = {
    'data_percent': [1, 0.5, 0.1, 0.01, 'uniform'],
    'loss': ['SSIM', 'MSE', 'LPIPS', 'LPIPS1', 'MSSIM', 'DISTS', 'NLPD'],
    'network': ['conv_big_z'],
    'seed': SEEDS,
}

# the same comparison on dcgan. on conv_big_z every DISTS run and most NLPD runs collapse
# (FINDINGS B3), so their budget curves cannot be read there; dcgan is conv_big_z plus
# BatchNorm and collapses in none of its runs. 1 and uniform are left out: the scaling factor
# there is 1, so those cells *are* the fixed-budget runs and already exist.
dcgan_budget = {
    'data_percent': [0.5, 0.1, 0.01],
    'loss': ['SSIM', 'MSE', 'LPIPS', 'DISTS'],
    'network': ['dcgan'],
    'seed': SEEDS,
}

# the losses dcgan had not been run on (pipeline_CIFAR_BIG.py 'dcgan_fill'), under equal steps
dcgan_fill_budget = dict(dcgan_budget, loss=['MSSIM', 'NLPD', 'LPIPS1'])

PARTITIONS = {
    'dcgan_fill': dcgan_fill_budget,
    'dcgan_fill_s42': dict(dcgan_fill_budget, seed=[42]),
    'dcgan_fill_s1': dict(dcgan_fill_budget, seed=[1]),
    'dcgan_fill_s2': dict(dcgan_fill_budget, seed=[2]),
    'dcgan': dcgan_budget,
    'dcgan_s42': dict(dcgan_budget, seed=[42]),
    'dcgan_s1': dict(dcgan_budget, seed=[1]),
    'dcgan_s2': dict(dcgan_budget, seed=[2]),
    'all': budget,
    'budget_s42': dict(budget, seed=[42]),
    'budget_s1': dict(budget, seed=[1]),
    'budget_s2': dict(budget, seed=[2]),
}


def go(runs):
    percept_loss.pipeline.generic.run(runs, CIFAR_AUTOENCODERS, epochs=30, batch_size=32,
                                      preload_data=True, dataset='CIFAR_10',
                                      n_validations=N_VALIDATIONS, props=PROPS,
                                      epoch_scaling='equal_steps')


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if name not in PARTITIONS:
        raise SystemExit(f'unknown partition {name!r}; pick from {sorted(PARTITIONS)}')
    go(PARTITIONS[name])
