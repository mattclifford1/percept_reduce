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


# dcgan at 50% data, fixed budget -- the partner the equal-steps dcgan grid is compared
# against (pipeline_CIFAR_BUDGET.py 'dcgan'). ARCH_SIZES skips 0.5, so only seed 42 existed and
# SSIM not at all; finished cells skip on done.json.
dcgan_half = {
    'data_percent': [0.5],
    'loss': ARCH_LOSSES,
    'network': ['dcgan'],
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
    'dcgan_half': [dcgan_half],
    'dcgan_half_s42': [dict(dcgan_half, seed=[42])],
    'dcgan_half_s1': [dict(dcgan_half, seed=[1])],
    'dcgan_half_s2': [dict(dcgan_half, seed=[2])],
    'all': [controls, headline, architectures],
}

# normalisation by default. dcgan is conv_big_z plus BatchNorm and collapses in none of its runs,
# so it becomes the main small backbone; these are the losses it had not been run on.
FILL_LOSSES = ['MSSIM', 'NLPD', 'LPIPS1']
dcgan_fill = {'data_percent': ALL_SIZES, 'loss': FILL_LOSSES, 'network': ['dcgan'], 'seed': SEEDS}
# GDN in place of BatchNorm (Balle, Laparra & Simoncelli 2016): a per-sample divisive
# normalisation from the image-coding literature, and the operation NLPD is built on
gdn = [dict(controls, network=['dcgan_gdn']), arch(['dcgan_gdn'])]
# ViT with a learning-rate warm-up; the RANDOM rows give the plots its untrained band
vit_wu = [dict(controls, network=['vit_wu']), arch(['vit_wu'])]

PARTITIONS.update({
    'dcgan_fill': [dcgan_fill],
    'gdn': gdn,
    'vit_wu': vit_wu,
    # two of the cells that collapse most on vit -- read these before running the other 46
    'vit_wu_pilot': [dict(arch(['vit_wu']), data_percent=[1], loss=['DISTS', 'SSIM'], seed=[42])],
})
for _s in SEEDS:
    PARTITIONS[f'dcgan_fill_s{_s}'] = [dict(dcgan_fill, seed=[_s])]
    PARTITIONS[f'gdn_s{_s}'] = [dict(block, seed=[_s]) for block in gdn]
    PARTITIONS[f'vit_wu_s{_s}'] = [dict(block, seed=[_s]) for block in vit_wu]


def go(runs):
    percept_loss.pipeline.generic.run(runs, CIFAR_AUTOENCODERS, epochs=30, batch_size=32,
                                      preload_data=True, dataset='CIFAR_10', validate_every=2)


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if name not in PARTITIONS:
        raise SystemExit(f'unknown partition {name!r}; pick from {sorted(PARTITIONS)}')
    for block in PARTITIONS[name]:
        go(block)
