import numpy as np
from math import sqrt
from numba import cuda


@cuda.jit("float32[:, :], float32[:, :], float32[:, :], float32[:, :], float32[:, :], float32[:, :], int64, int64, int64, int64")
def cuda_kern_block_matching_ncc(u, v, w, c, p, q, ys, xs, bs, sr):
    b = bs // 2
    bb = b + b + 1
    ofs = b + sr
    # image pixel coordinates (i, j)
    i, j = cuda.grid(2)
    if ofs <= i and i < ys - ofs and ofs <= j and j < xs - ofs:
        cmax = -1
        # find best correlation with varying block sizes
        for bi in range(b, 3, -1):
            src = p[i - bi:i + bi + 1, j - bi:j + bi + 1]
            # brute force search with search radius (sr)
            for n in range(-sr, sr + 1):
                jn = j + n
                for m in range(-sr, sr + 1):
                    im = i + m
                    tar = q[im - bi:im + bi + 1, jn - bi:jn + bi + 1]
                    # cost func (ncc)
                    smean = 0.0
                    tmean = 0.0
                    for ii in range(bi + bi + 1):
                        for jj in range(bi + bi + 1):
                            smean += src[ii, jj]
                            tmean += tar[ii, jj]
                    smean /= (bi + bi + 1) * (bi + bi + 1)
                    tmean /= (bi + bi + 1) * (bi + bi + 1)
                    cf = 0.0
                    sstdev = 0.0
                    tstdev = 0.0
                    for ii in range(bi + bi + 1):
                        for jj in range(bi + bi + 1):
                            cf += (src[ii, jj] - smean) * (tar[ii, jj] - tmean)
                            sstdev += (src[ii, jj] - smean) * (src[ii, jj] - smean)
                            tstdev += (tar[ii, jj] - tmean) * (tar[ii, jj] - tmean)
                    cf /= sqrt(sstdev)
                    cf /= sqrt(tstdev)
                    # update best correlation
                    if cf > cmax:
                        cmax = cf
                        nmax = n
                        mmax = m
                        wmax = bi
        c[i, j] = cmax
        u[i, j] = nmax
        v[i, j] = mmax
        w[i, j] = wmax


def block_matching_ncc(p, q, block_size, search_radius):
    ys, xs = p.shape
    # image dimensions have to match
    assert ys == q.shape[0]
    assert xs == q.shape[1]
    # minimum block size
    assert block_size > 8
    d_p = cuda.to_device(p.astype("float32"))
    d_q = cuda.to_device(q.astype("float32"))
    d_u = cuda.device_array((ys, xs), np.float32)
    d_v = cuda.device_array((ys, xs), np.float32)
    d_w = cuda.device_array((ys, xs), np.float32)
    d_c = cuda.device_array((ys, xs), np.float32)
    # adjust to your GPU
    nthreads = (16, 16)
    nblocksy = ys // nthreads[0] + 1
    nblocksx = xs // nthreads[0] + 1
    nblocks = (nblocksy, nblocksx)
    cuda_kern_block_matching_ncc[nblocks, nthreads](d_u, d_v, d_w, d_c, d_p, d_q, ys, xs, block_size, search_radius)
    c = d_c.copy_to_host()
    u = d_u.copy_to_host()
    v = d_v.copy_to_host()
    w = d_w.copy_to_host()
    mask = np.isnan(p) + np.isnan(q)
    u[mask] = np.nan
    v[mask] = np.nan
    w[mask] = np.nan
    c[mask] = np.nan
    return (u, v, w, c)


@cuda.jit("float32[:, :], float32[:, :], float32[:, :], float32[:, :], float32[:, :], float32[:, :], uint64[:], uint64[:], uint64, int64, int64")
def cuda_kern_block_matching_masked_ncc(u, v, w, c, p, q, ir, jr, nn, bs, sr):
    b = bs // 2
    bb = b + b + 1
    ofs = b + sr
    # image pixel coordinates (i, j) via (ir[k], jr[k])
    k = cuda.grid(1)
    if k < nn:
        i = ir[k]
        j = jr[k]
        cmax = -1
        # find best correlation with varying block sizes
        for bi in range(b, 3, -1):
            src = p[i - bi:i + bi + 1, j - bi:j + bi + 1]
            # brute force search with search radius (sr)
            for n in range(-sr, sr + 1):
                jn = j + n
                for m in range(-sr, sr + 1):
                    im = i + m
                    tar = q[im - bi:im + bi + 1, jn - bi:jn + bi + 1]
                    # cost func (ncc)
                    smean = 0.0
                    tmean = 0.0
                    for ii in range(bi + bi + 1):
                        for jj in range(bi + bi + 1):
                            smean += src[ii, jj]
                            tmean += tar[ii, jj]
                    smean /= (bi + bi + 1) * (bi + bi + 1)
                    tmean /= (bi + bi + 1) * (bi + bi + 1)
                    cf = 0.0
                    sstdev = 0.0
                    tstdev = 0.0
                    for ii in range(bi + bi + 1):
                        for jj in range(bi + bi + 1):
                            cf += (src[ii, jj] - smean) * (tar[ii, jj] - tmean)
                            sstdev += (src[ii, jj] - smean) * (src[ii, jj] - smean)
                            tstdev += (tar[ii, jj] - tmean) * (tar[ii, jj] - tmean)
                    cf /= sqrt(sstdev)
                    cf /= sqrt(tstdev)
                    # update best correlation
                    if cf > cmax:
                        cmax = cf
                        nmax = n
                        mmax = m
                        wmax = bi
        c[i, j] = cmax
        u[i, j] = nmax
        v[i, j] = mmax
        w[i, j] = wmax


def block_matching_masked_ncc(p, q, mask, block_size, search_radius):
    ys, xs = p.shape
    # image dimensions have to match
    assert ys == q.shape[0]
    assert xs == q.shape[1]
    # mask dimensions have to match
    assert ys == mask.shape[0]
    assert xs == mask.shape[1]    
    # minimum block size
    assert block_size > 8
    ms = mask.astype("bool").copy()
    offset = block_size // 2 + search_radius
    ms[:offset, :] = 0
    ms[:, :offset] = 0
    ms[-offset:, :] = 0
    ms[:, -offset:] = 0
    ir, jr = np.nonzero(~ms)
    d_ir = cuda.to_device(ir.astype("uint64"))
    d_jr = cuda.to_device(jr.astype("uint64"))
    d_p = cuda.to_device(p.astype("float32"))
    d_q = cuda.to_device(q.astype("float32"))
    d_u = cuda.device_array((ys, xs), np.float32)
    d_v = cuda.device_array((ys, xs), np.float32)
    d_w = cuda.device_array((ys, xs), np.float32)
    d_c = cuda.device_array((ys, xs), np.float32)
    # adjust to your GPU
    nthreads = 256
    nblocks = (len(ir) // nthreads) + 1
    cuda_kern_block_matching_masked_ncc[nblocks, nthreads](d_u, d_v, d_w, d_c, d_p, d_q, d_ir, d_jr, len(ir), block_size, search_radius)
    c = d_c.copy_to_host()
    u = d_u.copy_to_host()
    v = d_v.copy_to_host()
    w = d_w.copy_to_host()
    nan = np.isnan(p) + np.isnan(q)
    u[nan] = np.nan
    v[nan] = np.nan
    w[nan] = np.nan
    c[nan] = np.nan
    u[ms] = np.nan
    v[ms] = np.nan
    w[ms] = np.nan
    c[ms] = np.nan
    return (u, v, w, c)

