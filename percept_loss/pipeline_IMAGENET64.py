from percept_loss.networks import IMAGENET64_AUTOENCODERS
import percept_loss.pipeline.generic
# run configs
runs = {
    'data_percent': [1, 0.5, 0.1, 0.01, 'uniform'],
    # 'data_percent': ['uniform', 0.5, 0.1],
    'loss': ['SSIM', 'MSE', 'LPIPS', 'MSSIM', 'DISTS', 'NLPD'],
    # 'loss': ['DISTS', 'NLPD'],
    # 'network': ['standard'],
    'network': ['bigger_z'],
}

percept_loss.pipeline.generic.run(runs, IMAGENET64_AUTOENCODERS, epochs=30, batch_size=32, preload_data=True, dataset='IMAGENET64_VAL', validate_every=2)
