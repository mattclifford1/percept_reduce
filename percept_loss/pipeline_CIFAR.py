from percept_loss.networks import CIFAR_AUTOENCODERS
import percept_loss.pipeline.generic
# run configs
runs = {
        'data_percent': [1, 0.5, 0.1, 0.01, 'uniform'],
        # 'data_percent': ['uniform', 0.5, 0.1],
        'loss': ['SSIM', 'MSE', 'LPIPS', 'MSSIM', 'DISTS', 'NLPD'],
        # 'loss': ['DISTS', 'NLPD'],
        # 'network': ['conv_small_z', 'conv_bigger_z', 'conv_big_z'],
        'network': ['conv_big_z'],
    }

percept_loss.pipeline.generic.run(runs, CIFAR_AUTOENCODERS, epochs=30, batch_size=32, preload_data=True, dataset='CIFAR_10', validate_every=2)
    
