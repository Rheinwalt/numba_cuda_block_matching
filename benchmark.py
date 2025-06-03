import numpy as np
from numba import cuda
from time import time
from scipy.ndimage import gaussian_filter
from block_matching import block_matching_masked_ncc, block_matching_masked_ncc_int

# generate some input (p, q)
shape = np.array((1080, 1920))
p = np.random.normal(size=shape * 3)
p = gaussian_filter(p, sigma=100)
p -= p.min()
p /= p.max()
p *= 1000
p = p.astype("uint16")
q = p.copy()
q[10:-10, 20:-30] = p[12:-8, 17:-33]

block_size = 9
search_radius = 5
mask = np.ones(p.shape)
mask[3::7, 3::7] = 0

t0 = time()

# old float version
u, v, block_sizes, correlation = block_matching_masked_ncc(p, q, mask, block_size, search_radius)

t1 = time()

# new int version
ui, vi, corri = block_matching_masked_ncc_uint_nonzero(p, q, mask, block_size, search_radius)

t2 = time()

print("old version: t=%.2f s" % (t1-t0))
ul, lc = np.unique(u.ravel(), return_counts=True)
print("u:", ul[np.argsort(lc)[-2:]])
ul, lc = np.unique(v.ravel(), return_counts=True)
print("v:", ul[np.argsort(lc)[-2:]])

print("new version: t=%.2f s" % (t2-t1))
ul, lc = np.unique(ui.ravel(), return_counts=True)
print("u:", ul[np.argsort(lc)[-2:]])
ul, lc = np.unique(vi.ravel(), return_counts=True)
print("v:", ul[np.argsort(lc)[-2:]])
