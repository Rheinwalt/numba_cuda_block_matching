import numpy as np
from math import sqrt
from numba import cuda


@cuda.jit("int32[:, :], int32[:, :], int32[:, :], int32[:, :], float32[:, :], float32[:, :], int64, int64, int64, int64")
def cuda_kern_block_matching_ncc(u, v, w, c, p, q, ys, xs, bs, sr):
    b = bs // 2
    bb = b + b + 1
    ofs = b + sr
    # image pixel coordinates (i, j)
    i, j = cuda.grid(2)
    if i < ys and j < xs:
        c[i, j] = -999
        u[i, j] = -999
        v[i, j] = -999
        w[i, j] = -999
    if ofs <= i and i < ys - ofs and ofs <= j and j < xs - ofs:
        cmax = -10
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
                        wmax = bi + bi + 1
        c[i, j] = int(1000 * cmax)
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
    d_u = cuda.device_array((ys, xs), np.int32)
    d_v = cuda.device_array((ys, xs), np.int32)
    d_w = cuda.device_array((ys, xs), np.int32)
    d_c = cuda.device_array((ys, xs), np.int32)
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
    u[mask] = -999
    v[mask] = -999
    w[mask] = -999
    c[mask] = -999
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
            smean = 0.0
            for ii in range(bi + bi + 1):
                for jj in range(bi + bi + 1):
                    smean += src[ii, jj]
            smean /= (bi + bi + 1) * (bi + bi + 1)
            sstdev = 0.0
            for ii in range(bi + bi + 1):
                for jj in range(bi + bi + 1):
                    sstdev += (src[ii, jj] - smean) * (src[ii, jj] - smean)
            sstdev = sqrt(sstdev)
            # brute force search with search radius (sr)
            for n in range(-sr, sr + 1):
                jn = j + n
                for m in range(-sr, sr + 1):
                    im = i + m
                    tar = q[im - bi:im + bi + 1, jn - bi:jn + bi + 1]
                    # cost func (ncc)
                    tmean = 0.0
                    for ii in range(bi + bi + 1):
                        for jj in range(bi + bi + 1):
                            tmean += tar[ii, jj]
                    tmean /= (bi + bi + 1) * (bi + bi + 1)
                    cf = 0.0
                    tstdev = 0.0
                    for ii in range(bi + bi + 1):
                        for jj in range(bi + bi + 1):
                            cf += (src[ii, jj] - smean) * (tar[ii, jj] - tmean)
                            tstdev += (tar[ii, jj] - tmean) * (tar[ii, jj] - tmean)
                    cf /= sstdev
                    cf /= sqrt(tstdev)
                    # update best correlation
                    if cf > cmax:
                        cmax = cf
                        nmax = n
                        mmax = m
                        wmax = bi + bi + 1
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
    ms[:offset, :] = 1
    ms[:, :offset] = 1
    ms[-offset:, :] = 1
    ms[:, -offset:] = 1
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


@cuda.jit("int8[:, :, :], uint16[:, :], uint16[:, :], uint64[:], uint64[:], uint64, int64, int64")
def cuda_kern_block_matching_masked_ncc_uint_nonzero_fullc(c, p, q, ir, jr, nn, bs, sr):
    b = bs // 2
    bb = b + b + 1
    ofs = b + sr
    # image pixel coordinates (i, j) via (ir[k], jr[k])
    k = cuda.grid(1)
    if k < nn:
        i = ir[k]
        j = jr[k]
        src = p[i - b:i + b + 1, j - b:j + b + 1]
        smean = 0.0
        sn = 0
        # masking out zeros (NaNs)
        for ii in range(bb):
            for jj in range(bb):
                if src[ii, jj]:
                    smean += src[ii, jj]
                    sn += 1
        smean /= sn
        sstdev = 0.0
        for ii in range(bb):
            for jj in range(bb):
                if src[ii, jj]:
                    sstdev += (src[ii, jj] - smean) * (src[ii, jj] - smean)
        sstdev = sqrt(sstdev)
        # brute force search with search radius (sr)
        for m in range(-sr, sr + 1):
            im = i + m
            for n in range(-sr, sr + 1):
                jn = j + n
                tar = q[im - b:im + b + 1, jn - b:jn + b + 1]
                # cost func (ncc)
                tmean = 0.0
                tn = 0
                # masking out zeros (NaNs)
                for ii in range(bb):
                    for jj in range(bb):
                        if tar[ii, jj]:
                            tmean += tar[ii, jj]
                            tn += 1
                tmean /= tn
                cf = 0.0
                tstdev = 0.0
                for ii in range(bb):
                    for jj in range(bb):
                        if src[ii, jj] and tar[ii, jj]:
                            cf += (src[ii, jj] - smean) * (tar[ii, jj] - tmean)
                            tstdev += (tar[ii, jj] - tmean) * (tar[ii, jj] - tmean)
                cf /= sstdev
                cf /= sqrt(tstdev)
                # int8 -128 .. 127
                c[k, m+sr, n+sr] = int(127 * cf)


def block_matching_masked_ncc_uint_nonzero_fullc(p, q, mask, block_size, search_radius, nthreads_exp=10):
    ys, xs = p.shape
    sr = int(search_radius)
    srsr = sr + sr + 1

    # image dimensions have to match
    assert ys == q.shape[0]
    assert xs == q.shape[1]

    # mask dimensions have to match
    assert ys == mask.shape[0]
    assert xs == mask.shape[1]

    # mask layout
    ms = mask.astype(np.bool).copy()
    offset = block_size // 2 + sr
    ms[:offset, :] = True
    ms[:, :offset] = True
    ms[-offset:, :] = True
    ms[:, -offset:] = True
    ms += (p == 0)
    ms += (q == 0)
    ir, jr = np.nonzero(~ms)

    # cuda device arrays
    d_ir = cuda.to_device(ir.astype(np.uint64))
    d_jr = cuda.to_device(jr.astype(np.uint64))
    d_p = cuda.to_device(p.astype(np.uint16))
    d_q = cuda.to_device(q.astype(np.uint16))
    d_c = cuda.device_array((len(ir), srsr, srsr), np.int8)

    # GPU thread layout (adjust to your GPU)
    nthreads = 2**nthreads_exp
    nblocks = (len(ir) // nthreads) + 1

    cuda_kern_block_matching_masked_ncc_uint_nonzero_fullc[nblocks, nthreads](
            d_c, d_p, d_q, d_ir, d_jr, len(ir), block_size, search_radius)
    c = d_c.copy_to_host()
    return (c, ir, jr)


@cuda.jit("int8[:, :], int8[:, :], uint8[:, :], float32[:, :], uint16[:, :], uint16[:, :], uint64[:], uint64[:], uint64, int64, int64")
def cuda_kern_block_matching_masked_ncc_uint_nonzero(u, v, c, s, p, q, ir, jr, nn, bs, sr):
    b = bs // 2
    bb = b + b + 1
    ofs = b + sr
    # image pixel coordinates (i, j) via (ir[k], jr[k])
    k = cuda.grid(1)
    if k < nn:
        i = ir[k]
        j = jr[k]
        cmax = -1.0
        src = p[i - b:i + b + 1, j - b:j + b + 1]
        smean = 0.0
        sn = 0
        # masking out zeros (NaNs)
        for ii in range(bb):
            for jj in range(bb):
                if src[ii, jj]:
                    smean += src[ii, jj]
                    sn += 1
        smean /= sn
        sstdev = 0.0
        for ii in range(bb):
            for jj in range(bb):
                if src[ii, jj]:
                    sstdev += (src[ii, jj] - smean) * (src[ii, jj] - smean)
        sstdev = sqrt(sstdev)
        # brute force search with search radius (sr)
        for n in range(-sr, sr + 1):
            jn = j + n
            for m in range(-sr, sr + 1):
                im = i + m
                tar = q[im - b:im + b + 1, jn - b:jn + b + 1]
                # cost func (ncc)
                tmean = 0.0
                tn = 0
                # masking out zeros (NaNs)
                for ii in range(bb):
                    for jj in range(bb):
                        if tar[ii, jj]:
                            tmean += tar[ii, jj]
                            tn += 1
                tmean /= tn
                cf = 0.0
                tstdev = 0.0
                for ii in range(bb):
                    for jj in range(bb):
                        if src[ii, jj] and tar[ii, jj]:
                            cf += (src[ii, jj] - smean) * (tar[ii, jj] - tmean)
                            tstdev += (tar[ii, jj] - tmean) * (tar[ii, jj] - tmean)
                cf /= sstdev
                cf /= sqrt(tstdev)
                if cf > cmax:
                    cmax = cf
                    nmax = n
                    mmax = m
        c[i, j] = int(255 * cmax)
        u[i, j] = nmax
        v[i, j] = mmax
        s[i, j] = sstdev


def block_matching_masked_ncc_uint_nonzero(p, q, mask, block_size, search_radius, nthreads_exp=10):
    ys, xs = p.shape

    # image dimensions have to match
    assert ys == q.shape[0]
    assert xs == q.shape[1]

    # mask dimensions have to match
    assert ys == mask.shape[0]
    assert xs == mask.shape[1]

    # mask layout
    ms = mask.astype(np.bool).copy()
    offset = block_size // 2 + search_radius
    ms[:offset, :] = True
    ms[:, :offset] = True
    ms[-offset:, :] = True
    ms[:, -offset:] = True
    ms += (p == 0)
    ms += (q == 0)
    ir, jr = np.nonzero(~ms)

    # cuda device arrays
    d_ir = cuda.to_device(ir.astype(np.uint64))
    d_jr = cuda.to_device(jr.astype(np.uint64))
    d_p = cuda.to_device(p.astype(np.uint16))
    d_q = cuda.to_device(q.astype(np.uint16))
    d_u = cuda.device_array((ys, xs), np.int8)
    d_v = cuda.device_array((ys, xs), np.int8)
    d_c = cuda.device_array((ys, xs), np.uint8)
    d_s = cuda.device_array((ys, xs), np.float32)

    # GPU thread layout (adjust to your GPU)
    nthreads = 2**nthreads_exp
    nblocks = (len(ir) // nthreads) + 1

    cuda_kern_block_matching_masked_ncc_uint_nonzero[nblocks, nthreads](d_u, d_v, d_c, d_s,
        d_p, d_q, d_ir, d_jr, len(ir), block_size, search_radius)
    c = d_c.copy_to_host()
    u = d_u.copy_to_host()
    v = d_v.copy_to_host()
    s = d_s.copy_to_host()

    # NaN
    u[ms] = -128
    v[ms] = -128
    c[ms] = 0
    s[ms] = np.nan
    return (u, v, c, s)


@cuda.jit("int8[:, :, :], int8[:, :, :], uint8[:, :, :], uint16[:, :], uint16[:, :], uint64[:], uint64[:], uint64, int64, int64")
def cuda_kern_block_matching_masked_ncc_uint_nonzero_multiple(u, v, c, p, q, ir, jr, nn, bs, sr):
    b = bs // 2
    bb = b + b + 1
    ofs = b + sr
    nbc = 10
    # image pixel coordinates (i, j) via (ir[k], jr[k])
    k = cuda.grid(1)
    if k < nn:
        ncmax = cuda.local.array((nbc,), np.uint8)
        nnmax = cuda.local.array((nbc,), np.int8)
        nmmax = cuda.local.array((nbc,), np.int8)
        for ii in range(nbc):
            ncmax[ii] = 0
        i = ir[k]
        j = jr[k]
        cmax = -1.0
        src = p[i - b:i + b + 1, j - b:j + b + 1]
        smean = 0.0
        sn = 0
        # masking out zeros (NaNs)
        for ii in range(bb):
            for jj in range(bb):
                if src[ii, jj]:
                    smean += src[ii, jj]
                    sn += 1
        smean /= sn
        sstdev = 0.0
        for ii in range(bb):
            for jj in range(bb):
                if src[ii, jj]:
                    sstdev += (src[ii, jj] - smean) * (src[ii, jj] - smean)
        sstdev = sqrt(sstdev)
        # brute force search with search radius (sr)
        for n in range(-sr, sr + 1):
            jn = j + n
            for m in range(-sr, sr + 1):
                im = i + m
                tar = q[im - b:im + b + 1, jn - b:jn + b + 1]
                # cost func (ncc)
                tmean = 0.0
                tn = 0
                # masking out zeros (NaNs)
                for ii in range(bb):
                    for jj in range(bb):
                        if tar[ii, jj]:
                            tmean += tar[ii, jj]
                            tn += 1
                tmean /= tn
                cf = 0.0
                tstdev = 0.0
                for ii in range(bb):
                    for jj in range(bb):
                        if src[ii, jj] and tar[ii, jj]:
                            cf += (src[ii, jj] - smean) * (tar[ii, jj] - tmean)
                            tstdev += (tar[ii, jj] - tmean) * (tar[ii, jj] - tmean)
                cf /= sstdev
                cf /= sqrt(tstdev)
                icf = int(255 * cf)
                for ii in range(nbc):
                    if icf > ncmax[ii]:
                        for jj in range(nbc - 1, ii, -1):
                            ncmax[jj] = ncmax[jj - 1]
                            nnmax[jj] = nnmax[jj - 1]
                            nmmax[jj] = nmmax[jj - 1]
                        ncmax[ii] = icf
                        nnmax[ii] = n
                        nmmax[ii] = m
                        break
        for ii in range(nbc):
            c[i, j, ii] = ncmax[ii]
            u[i, j, ii] = nnmax[ii]
            v[i, j, ii] = nmmax[ii]


def block_matching_masked_ncc_uint_nonzero_multiple(p, q, mask, block_size, search_radius, nthreads_exp=10):
    ys, xs = p.shape
    nbest_corr = 10

    # image dimensions have to match
    assert ys == q.shape[0]
    assert xs == q.shape[1]

    # mask dimensions have to match
    assert ys == mask.shape[0]
    assert xs == mask.shape[1]

    # mask layout
    ms = mask.astype(np.bool).copy()
    offset = block_size // 2 + search_radius
    ms[:offset, :] = True
    ms[:, :offset] = True
    ms[-offset:, :] = True
    ms[:, -offset:] = True
    ms += (p == 0)
    ms += (q == 0)
    ir, jr = np.nonzero(~ms)

    # cuda device arrays
    d_ir = cuda.to_device(ir.astype(np.uint64))
    d_jr = cuda.to_device(jr.astype(np.uint64))
    d_p = cuda.to_device(p.astype(np.uint16))
    d_q = cuda.to_device(q.astype(np.uint16))
    d_u = cuda.device_array((ys, xs, nbest_corr), np.int8)
    d_v = cuda.device_array((ys, xs, nbest_corr), np.int8)
    d_c = cuda.device_array((ys, xs, nbest_corr), np.uint8)

    # GPU thread layout (adjust to your GPU)
    nthreads = 2**nthreads_exp
    nblocks = (len(ir) // nthreads) + 1

    cuda_kern_block_matching_masked_ncc_uint_nonzero_multiple[nblocks, nthreads](d_u, d_v, d_c,
        d_p, d_q, d_ir, d_jr, len(ir), block_size, search_radius)
    c = d_c.copy_to_host()
    u = d_u.copy_to_host()
    v = d_v.copy_to_host()

    # NaN
    for i in range(nbest_corr):
        u[ms, i] = -128
        v[ms, i] = -128
        c[ms, i] = 0
    return (u, v, c)
