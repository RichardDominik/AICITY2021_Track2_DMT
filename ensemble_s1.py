import numpy as np
import os

distmat_paths = ['./logs/stage1/lftd/dist_mat.npy']

# method 1
distmat = np.zeros((1103,31238))
for i in distmat_paths:
    distmat += np.load(i)

# method 2
# distmat = np.zeros((1103,31238))
# for i in distmat_paths:
#     distmat += (np.load(i) ** 2)
#
# distmat = np.sqrt(distmat)

sort_distmat_index = np.argsort(distmat, axis=1)
print(sort_distmat_index)
print('The shape of distmat is: {}'.format(distmat.shape))

save_path = './track_s1.txt'
with open(save_path,'w') as f:
    for item in sort_distmat_index:
        for i in range(99):
            f.write(str(item[i] + 1) + ' ')
        f.write(str(item[99] + 1) + '\n')

