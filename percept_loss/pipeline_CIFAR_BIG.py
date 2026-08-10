'''
the big re-run: everything on the current protocol, with checkpoints.

why re-run at all -- the committed grid cannot answer the question any more:
  * probe v2 (FINDINGS B13): standardising the probe moved the untrained MLP baseline from
    0.128 to 0.411. old and new numbers are not comparable, and the old runs have no
    checkpoints, so they cannot be re-probed -- only re-run.
  * n=1 (TODO T1.4): with one seed a loss effect and an init effect are indistinguishable.
    the DISTS collapse pattern rearranged completely when only the seed changed.
  * no checkpoints existed before, so every future probe change meant retraining. after this,
    percept_loss/testing/reprobe.py handles it in seconds.

order is deliberate: controls first (seconds each), then the headline grid, then the
architectures cheapest-first. killing this part-way still leaves the most valuable half done,
and re-running resumes -- done.json makes finished cells skip.

    python percept_loss/pipeline_CIFAR_BIG.py              # everything, sequentially
    python percept_loss/pipeline_CIFAR_BIG.py headline     # one partition (see PARTITIONS)

## Running partitions in parallel

**measured on this grid: the probes are 95% of wall time and the GPU sits at ~9%.** the nets
are tiny (0.15-15M params on 32x32); the real work is sklearn on CPU. so the way to use a spare
GPU is not a bigger batch -- it is more processes.

partitions are disjoint *by network*, which matters: `previously_done` is a check-then-write
with no lock, so two processes pointed at the same cell would both decide to run it. splitting
by network makes overlap impossible by construction. controls are safe alongside any partition
because RANDOM is not in ARCH_LOSSES.

bound the BLAS threads per process or they each grab all 32 cores and thrash:

    OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python percept_loss/pipeline_CIFAR_BIG.py resnet18 &
    OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python percept_loss/pipeline_CIFAR_BIG.py headline &
'''
import sys

from percept_loss.networks import CIFAR_AUTOENCODERS
import percept_loss.pipeline.generic

SEEDS = [42, 1, 2]
ALL_LOSSES = ['SSIM', 'MSE', 'LPIPS', 'LPIPS1', 'MSSIM', 'DISTS', 'NLPD']
ALL_SIZES = [1, 0.5, 0.1, 0.01, 'uniform']
ARCH_LOSSES = ['MSE', 'LPIPS', 'DISTS', 'SSIM']
ARCH_SIZES = [1, 0.1, 0.01, 'uniform']
ARCH_NETS = ['dcgan', 'vae', 'vit', 'resnet18']    # cheapest first; resnet18 is ~80x conv_big_z

# untrained controls -- the floor every trained number has to clear. no gradient steps, so one
# data_percent is enough and the whole block costs a couple of minutes.
controls = {
    'data_percent': [1],
    'loss': ['RANDOM'],
    'network': ['conv_big_z'] + ARCH_NETS,
    'seed': SEEDS,
}

# the headline grid, on the same axes as the committed one so the tables line up
headline = {
    'data_percent': ALL_SIZES,
    'loss': ALL_LOSSES,
    'network': ['conv_big_z'],
    'seed': SEEDS,
}

# the literature backbones, cut down: 4 losses x 4 sizes x 3 seeds each
architectures = {
    'data_percent': ARCH_SIZES,
    'loss': ARCH_LOSSES,
    'network': ARCH_NETS,
    'seed': SEEDS,
}


def arch(nets):
    return dict(architectures, network=nets)


# disjoint by network, so any subset of these can run at the same time
PARTITIONS = {
    'controls': [controls],
    'headline': [headline],                     # conv_big_z, 105 cells
    # headline is 105 cells against 48 for each architecture, so it is the long pole if run as
    # one process. seeds are part of the run directory, so splitting on them is disjoint too.
    'headline_s42': [dict(headline, seed=[42])],
    'headline_s1': [dict(headline, seed=[1])],
    'headline_s2': [dict(headline, seed=[2])],
    'dcgan': [arch(['dcgan'])],
    'vae': [arch(['vae'])],
    'vit': [arch(['vit'])],
    'resnet18': [arch(['resnet18'])],           # slowest -- start it first
    'all': [controls, headline, architectures],
}


def go(runs):
    percept_loss.pipeline.generic.run(runs, CIFAR_AUTOENCODERS, epochs=30, batch_size=32,
                                      preload_data=True, dataset='CIFAR_10', validate_every=2)


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if name not in PARTITIONS:
        raise SystemExit(f'unknown partition {name!r}; pick from {sorted(PARTITIONS)}')
    for block in PARTITIONS[name]:
        go(block)
