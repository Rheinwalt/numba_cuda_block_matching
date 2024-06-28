import numpy as np
from numba import cuda


@cuda.jit("float32[:, :], float32[:, :], float32[:, :], float32[:, :], float32[:, :], float32[:, :], int64, int64, int64, int64")
def cuda_kern_block_matching_ncc(u, v, w, c, p, q, ys, xs, bs, sr):
    b = bs // 2
    bb = b + b + 1
    ofs = b + sr
    i, j = cuda.grid(2)
    if ofs <= i and i < ys - ofs and ofs <= j and j < xs - ofs:
        cmax = -1
        for bi in range(b, 3, -1):
            src = p[i - bi:i + bi + 1, j - bi:j + bi + 1]
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
    assert ys == q.shape[0]
    assert xs == q.shape[1]
    assert block_size > 8
    d_p = cuda.to_device(p.astype("float32"))
    d_q = cuda.to_device(q.astype("float32"))
    d_u = cuda.device_array((ys, xs), np.float32)
    d_v = cuda.device_array((ys, xs), np.float32)
    d_w = cuda.device_array((ys, xs), np.float32)
    d_c = cuda.device_array((ys, xs), np.float32)
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

