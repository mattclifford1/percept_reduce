from percept_loss.networks import IMAGENET64_AUTOENCODERS
import percept_loss.pipeline.generic
# run configs
runs = {
    'data_percent': [1, 0.5, 0.1, 0.01, 'uniform'],
    # 'data_percent': ['uniform', 0.5, 0.1],
    'loss': ['SSIM', 'MSE', 'LPIPS', 'MSSIM', 'DISTS', 'NLPD'],
    # 'loss': ['DISTS', 'NLPD'],
    # 'network': ['conv_small_z', 'conv_bigger_z', 'conv_big_z'],
    'network': ['standard'],
}

percept_loss.pipeline.generic.run(runs, IMAGENET64_AUTOENCODERS, epochs=30, batch_size=32, preload_data=True, dataset='IMAGENET64_TRAIN', validate_every=1)
