import numpy as np
import os


# Results from paper after stage 1
# ResNet101-IBN-a = 54.0 = 0.98
# ResNet101-IBN-a-v1 = 54.9 = 1
# ResNet101-IBN-a-v2 = 54.3 = 0.99
# ResNext101-IBN-a = 50.6 = 0.92
# ResNest101 = 51.2 = 0.93
# SeResNet101-IBN-a = 51.6 = 0.94
# DenseNet169-IBN-a = 51.2 = 0.93
# TransReID = 47.5 = 0.87

# Added Swin
# Swin = 47.5 = 0.7


distmat_paths = {
        './logs/stage2/resnext101a_384/v1/dist_mat.npy': 0.92,
        './logs/stage2/resnext101a_384/v2/dist_mat.npy': 0.92,

        './logs/stage2/101a_384/v1/dist_mat.npy': 1,
        './logs/stage2/101a_384/v2/dist_mat.npy': 1,

        './logs/stage2/101a_384_recrop/v1/dist_mat.npy': 1,
        './logs/stage2/101a_384_recrop/v2/dist_mat.npy': 1,

        './logs/stage2/101a_384_spgan/v1/dist_mat.npy': 1,
        './logs/stage2/101a_384_spgan/v2/dist_mat.npy': 1,

        './logs/stage2/densenet169a_384/v1/dist_mat.npy': 0.93,
        './logs/stage2/densenet169a_384/v2/dist_mat.npy': 0.93,

        './logs/stage2/s101_384/v1/dist_mat.npy': 0.93,
        './logs/stage2/s101_384/v2/dist_mat.npy': 0.93,

        './logs/stage2/se_resnet101a_384/v1/dist_mat.npy': 0.94,
        './logs/stage2/se_resnet101a_384/v2/dist_mat.npy': 0.94,

        './logs/stage2/transreid_256/v1/dist_mat.npy': 0.87,
        './logs/stage2/transreid_256/v2/dist_mat.npy': 0.87,

        'logs/stage2/swin_transformer/swin_transformer_224_80_epochs/v1/dist_mat.npy': 0.75,
        'logs/stage2/swin_transformer/swin_transformer_224_80_epochs/v2/dist_mat.npy': 0.75,

	    # './logs/stage2/swin_transformer/v1/dist_mat.npy': 0.75,
	    # './logs/stage2/swin_transformer/v2/dist_mat.npy': 0.75,
        #
        # './logs/stage2/swin_transformer_spgan/v1/dist_mat.npy': 0.75,
        # './logs/stage2/swin_transformer_spgan/v2/dist_mat.npy': 0.75,

        # './logs/stage2/swin_transformer_384/v1/pth29/dist_mat.npy': 0.85,
        # './logs/stage2/swin_transformer_384/v2/pth29/dist_mat.npy': 0.85,
}

distmat = np.zeros((1103,31238))
for i in distmat_paths:
    distmat += (np.load(i) * distmat_paths[i])

sort_distmat_index = np.argsort(distmat, axis=1)
print(sort_distmat_index)
print('The shape of distmat is: {}'.format(distmat.shape))

save_path = './track2.txt'
with open(save_path,'w') as f:
    for item in sort_distmat_index:
        for i in range(99):
            f.write(str(item[i] + 1) + ' ')
        f.write(str(item[99] + 1) + '\n')

