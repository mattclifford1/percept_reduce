'''
architecture sweep -- the loss axis run on the literature backbones.

kept separate from pipeline_CIFAR.py so the committed loss x data-size grid stays exactly as it
is (and so this can be started while that one is still running -- different network name means
a different saves/ directory, no collision).

the loss list is cut to four: MSE as the pixel baseline, LPIPS as the perceptual one the whole
hypothesis rests on, DISTS because whether it still collapses on a normalised backbone is the
cheap test of FINDINGS B3, and SSIM as a second non-deep perceptual metric. running all seven
losses x five data sizes x five networks is 175 cells, which is not worth it before the four-loss
version says whether the architecture axis moves anything.
'''
from percept_loss.networks import CIFAR_AUTOENCODERS
import percept_loss.pipeline.generic

# the untrained-encoder control. no gradient steps, so data_percent is irrelevant -- one cell
# per network. run this first: it is seconds per network and it sets the floor every trained
# number has to beat.
control = {
    'data_percent': [1],
    'loss': ['RANDOM'],
    'network': ['conv_big_z', 'dcgan', 'resnet18', 'vit', 'vae'],
}

# the sweep. resnet18_thin is left out -- add it if resnet18 turns out to be too slow.
runs = {
    'data_percent': [1, 0.1, 0.01, 'uniform'],
    'loss': ['MSE', 'LPIPS', 'DISTS', 'SSIM'],
    'network': ['dcgan', 'resnet18', 'vit', 'vae'],
}

if __name__ == '__main__':
    percept_loss.pipeline.generic.run(control, CIFAR_AUTOENCODERS, epochs=30, batch_size=32,
                                      preload_data=True, dataset='CIFAR_10', validate_every=2)
    percept_loss.pipeline.generic.run(runs, CIFAR_AUTOENCODERS, epochs=30, batch_size=32,
                                      preload_data=True, dataset='CIFAR_10', validate_every=2)
