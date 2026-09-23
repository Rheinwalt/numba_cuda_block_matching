import numpy as np
from math import sqrt
from numba import cuda


@cuda.jit(device=True, inline=True)
def masked_pearson_uint16(a, b, ia, ja, ib, jb, half_block, min_valid):
    """
    Pearson correlation between blocks centered at (ia, ja) in a
    and (ib, jb) in b. Only locations where a and b are non-zero
    are used.

    Returns
    -------
    r    : Pearson correlation; -2 means invalid
    ss_a : sum of squared deviations in a
    ss_b : sum of squared deviations in b
    n    : number of common-valid pixels
    """
    suma = np.float32(0.0)
    sumb = np.float32(0.0)
    n = 0

    for di in range(-half_block, half_block + 1):
        for dj in range(-half_block, half_block + 1):
            av = a[ia + di, ja + dj]
            bv = b[ib + di, jb + dj]
            if av != 0 and bv != 0:
                suma += np.float32(av)
                sumb += np.float32(bv)
                n += 1

    if n < min_valid:
        return np.float32(-2.0), np.float32(0.0), np.float32(0.0), n

    amean = suma / np.float32(n)
    bmean = sumb / np.float32(n)

    ss_a = np.float32(0.0)
    ss_b = np.float32(0.0)
    ss_ab = np.float32(0.0)

    for di in range(-half_block, half_block + 1):
        for dj in range(-half_block, half_block + 1):
            av = a[ia + di, ja + dj]
            bv = b[ib + di, jb + dj]
            if av != 0 and bv != 0:
                da = np.float32(av) - amean
                db = np.float32(bv) - bmean
                ss_a += da * da
                ss_b += db * db
                ss_ab += da * db

    if ss_a <= 0.0 or ss_b <= 0.0:
        return np.float32(-2.0), ss_a, ss_b, n

    r = ss_ab / sqrt(ss_a * ss_b)
    return r, ss_a, ss_b, n


@cuda.jit
def cuda_kern_block_matching_masked_ncc_uint_nonzero_fb(
        u, v, cmax8, cmean8, cstd8, sstd16, tstd16, efb2,
        p, q, ir, jr, nn, bs, sr, min_valid):
    k = cuda.grid(1)
    if k >= nn:
        return

    ys = p.shape[0]
    xs = p.shape[1]
    b = bs // 2
    i = ir[k]
    j = jr[k]
    u[i, j] = -128
    v[i, j] = -128
    cmax8[i, j] = 0
    cmean8[i, j] = 0
    cstd8[i, j] = 0
    sstd16[i, j] = 0
    tstd16[i, j] = 0

    # UINT16_MAX means that no usable backward match was found.
    efb2[i, j] = 65535

    # forward p -> q
    best_c = np.float32(-2.0)
    best_n = 0
    best_m = 0

    best_sstd = np.float32(0.0)
    best_tstd = np.float32(0.0)

    csum = np.float32(0.0)
    csum2 = np.float32(0.0)
    nc = 0

    for n in range(-sr, sr + 1):
        jt = j + n
        for m in range(-sr, sr + 1):
            it = i + m

            cf, ss_s, ss_t, nv = masked_pearson_uint16(
                p, q,
                i, j,
                it, jt,
                b,
                min_valid
            )

            if cf > -1.5:
                csum += cf
                csum2 += cf * cf
                nc += 1
                if cf > best_c:
                    best_c = cf
                    best_n = n
                    best_m = m

                    # real standard deviations for winning pair
                    best_sstd = sqrt(ss_s / np.float32(nv))
                    best_tstd = sqrt(ss_t / np.float32(nv))

    # no usable forward candidate
    if nc == 0:
        return

    # correlation stats
    cmean = csum / np.float32(nc)

    cvar = csum2 / np.float32(nc) - cmean * cmean

    # floating-point roundoff
    if cvar < 0.0:
        cvar = np.float32(0.0)

    cstd = sqrt(cvar)

    u[i, j] = best_n
    v[i, j] = best_m

    # negative correlations are not interesting here
    # store 0...1 as 0...255.

    cc = best_c
    if cc < 0.0:
        cc = np.float32(0.0)
    if cc > 1.0:
        cc = np.float32(1.0)

    cmax8[i, j] = int(cc * 255.0 + 0.5)

    cm = cmean
    if cm < 0.0:
        cm = np.float32(0.0)
    if cm > 1.0:
        cm = np.float32(1.0)

    cmean8[i, j] = int(cm * 255.0 + 0.5)

    cs = cstd
    if cs < 0.0:
        cs = np.float32(0.0)
    if cs > 1.0:
        cs = np.float32(1.0)

    cstd8[i, j] = int(cs * 255.0 + 0.5)

    # stds remain in original input-value units
    if best_sstd > 65535.0:
        sstd16[i, j] = 65535
    else:
        sstd16[i, j] = int(best_sstd + 0.5)

    if best_tstd > 65535.0:
        tstd16[i, j] = 65535
    else:
        tstd16[i, j] = int(best_tstd + 0.5)

    # q -> p
    # Start at endpoint of best forward match.
    iq = i + best_m
    jq = j + best_n

    back_c = np.float32(-2.0)
    back_n = 0
    back_m = 0

    for n in range(-sr, sr + 1):
        jp = jq + n
        for m in range(-sr, sr + 1):
            ip = iq + m

            # reverse search can move farther toward an edge.
            if ip - b < 0:
                continue
            if ip + b >= ys:
                continue
            if jp - b < 0:
                continue
            if jp + b >= xs:
                continue

            cf, ss1, ss2, nv = masked_pearson_uint16(
                q, p,
                iq, jq,
                ip, jp,
                b,
                min_valid
            )

            if cf > back_c:
                back_c = cf
                back_n = n
                back_m = m

    # forward/backward consistency
    if back_c > -1.5:

        du = best_n + back_n
        dv = best_m + back_m

        # squared displacement, no sqrt on device
        e2 = du * du + dv * dv

        if e2 > 65535:
            e2 = 65535

        efb2[i, j] = e2


def block_matching_masked_ncc_uint_nonzero_fb(
        p, q, mask, block_size, search_radius,
        min_valid_frac=0.5, nthreads_exp=10):
    if p.ndim != 2 or q.ndim != 2 or mask.ndim != 2:
        raise ValueError("p, q, and mask must all be 2-D arrays")

    ys, xs = p.shape
    if q.shape != p.shape or mask.shape != p.shape:
        raise ValueError("p, q, and mask must have the same shape")

    bs = int(block_size)
    sr = int(search_radius)
    if bs != block_size or bs <= 0 or bs % 2 == 0:
        raise ValueError("block_size must be a positive odd integer")
    if sr != search_radius or not 0 < sr <= 127:
        raise ValueError(
            "search_radius must be an integer in [1, 127] because u and v "
            "are stored as int8 (-128 is reserved for nodata)"
        )
    if not 0.0 < min_valid_frac <= 1.0:
        raise ValueError("min_valid_frac must be in (0, 1]")
    if int(nthreads_exp) != nthreads_exp or not 0 <= nthreads_exp <= 10:
        raise ValueError("nthreads_exp must be an integer in [0, 10]")
    nthreads_exp = int(nthreads_exp)

    # actual odd block width used by kernel
    b = bs // 2
    bb = 2 * b + 1
    min_valid = int(np.ceil(min_valid_frac * bb * bb))

    # source centers to process
    ms = mask.astype(bool).copy()

    offset = b + sr

    ms[:offset, :] = True
    ms[:, :offset] = True
    ms[-offset:, :] = True
    ms[:, -offset:] = True

    # zero == NaN convention
    ms |= (p == 0)
    ms |= (q == 0)

    ir, jr = np.nonzero(~ms)

    # input arrays
    d_ir = cuda.to_device(ir.astype(np.int32))
    d_jr = cuda.to_device(jr.astype(np.int32))
    d_p = cuda.to_device(p.astype(np.uint16))
    d_q = cuda.to_device(q.astype(np.uint16))

    # output
    d_u = cuda.device_array((ys, xs), np.int8)
    d_v = cuda.device_array((ys, xs), np.int8)
    d_cmax = cuda.device_array((ys, xs), np.uint8)
    d_cmean = cuda.device_array((ys, xs), np.uint8)
    d_cstd = cuda.device_array((ys, xs), np.uint8)
    d_sstd = cuda.device_array((ys, xs), np.uint16)
    d_tstd = cuda.device_array((ys, xs), np.uint16)
    d_efb2 = cuda.device_array((ys, xs), np.uint16)

    nthreads = 2**nthreads_exp
    nblocks = len(ir) // nthreads + 1

    cuda_kern_block_matching_masked_ncc_uint_nonzero_fb[
        nblocks, nthreads
    ](
        d_u, d_v,
        d_cmax, d_cmean, d_cstd,
        d_sstd, d_tstd,
        d_efb2,
        d_p, d_q,
        d_ir, d_jr, len(ir),
        bs, sr,
        min_valid
    )

    u = d_u.copy_to_host()
    v = d_v.copy_to_host()
    cmax8 = d_cmax.copy_to_host()
    cmean8 = d_cmean.copy_to_host()
    cstd8 = d_cstd.copy_to_host()
    sstd = d_sstd.copy_to_host()
    tstd = d_tstd.copy_to_host()
    efb2 = d_efb2.copy_to_host()

    # mark unprocessed pixels
    u[ms] = -128
    v[ms] = -128

    cmax8[ms] = 0
    cmean8[ms] = 0
    cstd8[ms] = 0

    sstd[ms] = 0
    tstd[ms] = 0
    efb2[ms] = 0

    return u, v, cmax8, cmean8, cstd8, sstd, tstd, efb2
