"""Numba kernels for exact Metropolis-Hastings chains on perfect matchings of K_{n,n}.

Target: the posterior of the Gaussian planted matching model

    P(M | W)  propto  exp( mu * <Pi_M, W> ),   W = mu Pi_{M*} + Z,  mu^2 = C log n.

All moves are cyclic rotations of the current matching: rows r_0..r_{L-1}
pass their columns along a cycle,  r_l -> pi(r_{l+1}).  L = 2 is the swap of
the write-up.  Three kernels:

* ``uniform``  (``_uniform_chain``) - the swap chain exactly as in the write-up:
  (i, j) uniform, accept w.p. min(1, exp(mu * Delta)).

* ``rotation`` (``_informed_chains``) - informed rotations of length 2..5 with
  forced closure; row r_l proposes its new column from
  q_i(j) propto exp(mu W_ij + b_j)  via O(1) alias tables.

* ``walk``     (``_walk_chains``, default) - ejection chains: each displaced row
  draws a new column from q until the walk returns to the start column, so
  alternating cycles of any length <= max_len are proposed.  The MH ratio is
  prod (1 - q_old)/(1 - q_new) up to the eps / capped-rejection corrections,
  i.e. the move is almost rejection-free.  Long alternating cycles are exactly
  how posterior samples differ from M* and from each other near C = 2..4, which
  is why this kernel mixes orders of magnitude faster than 2-swaps there.

The column prices b_j (from a few Sinkhorn sweeps, i.e. a mean-field
approximation of the marginals) cancel in every cyclic rotation, so they only
change the efficiency, never the target.  exp(mu*Delta) is evaluated as a
product of ratios of Q-entries, so the hot loops read only Q (float64) and the
packed alias table: ~2 random cache lines per row touched.

Exactness of every kernel (including annealing, eps-mixing, capped rejection)
is checked against brute-force enumeration in ``tests/``.

Bookkeeping
-----------
* pi and pi^{-1} are both kept, so moves cost O(L).
* Edge marginals p_e = P(e in M | W) are *exact* ergodic time averages: every
  row remembers when its current edge was entered and adds the holding time to
  an integer counter only when the edge is left.
* |M_t cap M*| and <Pi_{M_t}, W> are updated in O(L) per accepted move; their
  time averages estimate n(1 - mmse_n) and (Nishimori) n*mu.
* Random numbers come from an inline xoshiro256** stream per chain.
* R chains on the same instance are advanced in lock-step so the CPU overlaps
  their DRAM-bound memory accesses.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(cache=True)
def seed_numba(seed):
    np.random.seed(seed)


@njit(cache=True)
def _overlap(perm, sigma):
    s = 0
    for i in range(perm.shape[0]):
        if perm[i] == sigma[i]:
            s += 1
    return s


# ---------------------------------------------------------------------------
# Proposal tables  q_i(j) propto exp(mu * W_ij)
# ---------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def sinkhorn_log(W, mu, iters):
    """Column log-prices b such that  K_ij = exp(mu W_ij + a_i + b_j)  is
    (approximately) doubly stochastic: a mean-field approximation of the
    posterior marginals.  Any b gives an exact chain (b cancels in the MH
    ratio of a cyclic rotation); a good b makes the proposal far better.
    """
    n = W.shape[0]
    a = np.zeros(n)
    b = np.zeros(n)
    for _ in range(iters):
        for i in prange(n):
            m = -np.inf
            for j in range(n):
                v = mu * W[i, j] + b[j]
                if v > m:
                    m = v
            s = 0.0
            for j in range(n):
                s += math.exp(mu * W[i, j] + b[j] - m)
            a[i] = -(m + math.log(s))
        for j in prange(n):
            m = -np.inf
            for i in range(n):
                v = mu * W[i, j] + a[i]
                if v > m:
                    m = v
            s = 0.0
            for i in range(n):
                s += math.exp(mu * W[i, j] + a[i] - m)
            b[j] = -(m + math.log(s))
    return b


@njit(cache=True, parallel=True)
def build_tables(W, scale, b):
    """Row-softmax Q[i, j] propto exp(scale * W_ij + b_j) (float64) and packed
    Vose alias tables AL[i, c] = (prob, alias); also returns log Z_i.

    The alias index is stored as float32, which is exact for n < 2^24.
    """
    n = W.shape[0]
    Q = np.empty((n, n), dtype=np.float64)
    AL = np.empty((n, n, 2), dtype=np.float32)
    logZ = np.empty(n, dtype=np.float64)
    for i in prange(n):
        m = -np.inf
        for j in range(n):
            v = scale * W[i, j] + b[j]
            if v > m:
                m = v
        p = np.empty(n, dtype=np.float64)
        s = 0.0
        for j in range(n):
            p[j] = math.exp(scale * W[i, j] + b[j] - m)
            s += p[j]
        logZ[i] = m + math.log(s)
        for j in range(n):
            Q[i, j] = p[j] / s
        small = np.empty(n, dtype=np.int64)
        large = np.empty(n, dtype=np.int64)
        ns = 0
        nl = 0
        for j in range(n):
            p[j] = p[j] * n / s
            if p[j] < 1.0:
                small[ns] = j
                ns += 1
            else:
                large[nl] = j
                nl += 1
        while ns > 0 and nl > 0:
            ns -= 1
            sm = small[ns]
            lg = large[nl - 1]
            AL[i, sm, 0] = p[sm]
            AL[i, sm, 1] = lg
            p[lg] = (p[lg] + p[sm]) - 1.0
            if p[lg] < 1.0:
                nl -= 1
                small[ns] = lg
                ns += 1
        while nl > 0:
            nl -= 1
            AL[i, large[nl], 0] = 1.0
            AL[i, large[nl], 1] = large[nl]
        while ns > 0:
            ns -= 1
            AL[i, small[ns], 0] = 1.0
            AL[i, small[ns], 1] = small[ns]
    return Q, AL, logZ


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Inline xoshiro256** generator (one independent stream per chain).
# ~1.5 ns per 64-bit draw vs ~10 ns for numba's np.random calls.
# ---------------------------------------------------------------------------
_U32 = np.uint64(0xFFFFFFFF)
_S32 = np.uint64(32)
_INV32 = 1.0 / 4294967296.0
_INV53 = 1.0 / 9007199254740992.0


@njit(cache=True, inline="always")
def _rotl(x, k):
    return (x << np.uint64(k)) | (x >> np.uint64(64 - k))


@njit(cache=True, inline="always")
def _next(st, r):
    s0 = st[r, 0]
    s1 = st[r, 1]
    s2 = st[r, 2]
    s3 = st[r, 3]
    result = _rotl(s1 * np.uint64(5), 7) * np.uint64(9)
    t = s1 << np.uint64(17)
    s2 ^= s0
    s3 ^= s1
    s1 ^= s2
    s0 ^= s3
    s2 ^= t
    s3 = _rotl(s3, 45)
    st[r, 0] = s0
    st[r, 1] = s1
    st[r, 2] = s2
    st[r, 3] = s3
    return result


@njit(cache=True, inline="always")
def _bounded(x32, nn, st, r):
    """Exactly uniform integer in [0, n) from a 32-bit word (Lemire 2019)."""
    m = x32 * nn
    lo = m & _U32
    if lo < nn:
        thresh = (np.uint64(4294967296) - nn) % nn
        while lo < thresh:
            x32 = _next(st, r) >> _S32
            m = x32 * nn
            lo = m & _U32
    return np.int64(m >> _S32)


@njit(cache=True, inline="always")
def _unif(st, r):
    return np.float64(_next(st, r) >> np.uint64(11)) * _INV53


def make_rng_state(seed_seq, R):
    """(R, 4) uint64 xoshiro states from a numpy SeedSequence (never all-zero)."""
    st = np.asarray(seed_seq.generate_state(4 * R, dtype=np.uint64)).reshape(R, 4)
    st[:, 0] |= np.uint64(1)
    return np.ascontiguousarray(st)


@njit(cache=True)
def _informed_chains(Q, AL, mu, eps16, len_cdf, perms, sigma, n_burn, n_sample,
                     n_anneal, group, counts, n_trace, rng):
    """R independent chains, advanced in lock-step on one instance, with
    informed k-cycle rotation moves (k = 2 is the swap move of the write-up).

    Move.  Draw a length L (P(L = l) from ``len_cdf``), a row r_0 uniformly, and
    for l = 0..L-2 a column c_l ~ qt_{r_l} where qt = (1-eps) q + eps/n is the
    row-softmax / uniform mixture; set r_{l+1} = pi^{-1}(c_l).  If the rows
    r_0..r_{L-1} are distinct, propose the cyclic rotation
        r_l -> pi(r_{l+1})  (l < L-1),     r_{L-1} -> pi(r_0).
    The same new matching is generated from any of the L cyclic starting rows,
    so with a_l = qt_{r_l}(pi(r_{l+1})) and b_l = qt_{r_l}(pi(r_l)):
        fwd = P(L)/n * prod(a) * sum(1/a),   rev = P(L)/n * prod(b) * sum(1/b),
    and exp(mu*Delta) = prod_l q_{r_l}(pi(r_{l+1})) / q_{r_l}(pi(r_l)).
    Non-simple paths are rejected (they are self-loops of the kernel).
    For L = 2 this is exactly the informed swap.

    perms : (R, n) int64, modified in place.
    eps16 : int, uniform-mixing weight eps = eps16 / 2^16 (exactly what the
        16-bit coins realise).
    len_cdf : (Lmax-1,) float64 cumulative distribution of L = 2..Lmax (Lmax<=5).
    n_anneal : (R,) annealing length per chain (0 = start at the exact mu).
    group : (R,) int64; chain r adds its holding times to counts[group[r]].
    counts : (G, n, n) integer array, or shape (G, 1, 1) to skip.
    rng : (R, 4) uint64 xoshiro256** states (modified in place).
    """
    R, n = perms.shape
    nn = np.uint64(n)
    eps = eps16 / 65536.0
    e16 = np.uint64(eps16)
    m16 = np.uint64(0xFFFF)
    unif_q = eps / n
    i_mass = 1.0 - eps
    LM = len_cdf.shape[0] + 1
    rows = np.empty(LM, dtype=np.int64)

    invs = np.empty((R, n), dtype=np.int64)
    for r in range(R):
        for i in range(n):
            invs[r, perms[r, i]] = i
    track = counts.shape[1] == n
    start = np.zeros((R, n), dtype=np.int64)
    ov = np.zeros(R, dtype=np.int64)
    energy = np.zeros(R, dtype=np.float64)
    for r in range(R):
        ov[r] = _overlap(perms[r], sigma)
        for i in range(n):
            energy[r] += math.log(Q[i, perms[r, i]]) / mu

    total = n_burn + n_sample
    trace_ov = np.zeros((R, n_trace), dtype=np.int64)
    trace_en = np.zeros((R, n_trace), dtype=np.float64)
    trace_every = max(total // max(n_trace, 1), 1)
    tr = 0
    ov_sum = np.zeros(R, dtype=np.int64)
    ov_sq_sum = np.zeros(R, dtype=np.float64)
    en_sum = np.zeros(R, dtype=np.float64)
    n_acc = np.zeros(R, dtype=np.int64)

    for t in range(total):
        sampling = t >= n_burn
        for r in range(R):
            expo = 1.0
            if t < n_anneal[r]:
                expo = math.sqrt((t + 1.0) / n_anneal[r])  # C_t linear in t
            x = _next(rng, r)
            r0 = _bounded(x >> _S32, nn, rng, r)
            uL = np.float64(x & _U32) * _INV32
            L = 2
            while L <= LM and uL >= len_cdf[L - 2]:
                L += 1
            if L > LM:
                L = LM
            coins = _next(rng, r)
            rows[0] = r0
            ok = True
            for l in range(L - 1):
                y = _next(rng, r)
                ri = rows[l]
                if (coins & m16) >= e16:
                    c = _bounded(y >> _S32, nn, rng, r)
                    if np.float64(y & _U32) * _INV32 >= AL[ri, c, 0]:
                        c = np.int64(AL[ri, c, 1])
                else:
                    c = _bounded(y >> _S32, nn, rng, r)
                coins >>= np.uint64(16)
                nxt = invs[r, c]
                for m in range(l + 1):
                    if rows[m] == nxt:
                        ok = False
                        break
                if not ok:
                    break
                rows[l + 1] = nxt
            if ok:
                # forward / reverse proposal mass and Boltzmann ratio
                pa = 1.0
                sa = 0.0
                pb = 1.0
                sb = 0.0
                boltz = 1.0
                for l in range(L):
                    ri = rows[l]
                    rn = rows[l + 1] if l + 1 < L else rows[0]
                    qa = Q[ri, perms[r, rn]]
                    qb = Q[ri, perms[r, ri]]
                    boltz *= qa / qb
                    ta = i_mass * qa + unif_q
                    tb = i_mass * qb + unif_q
                    pa *= ta
                    sa += 1.0 / ta
                    pb *= tb
                    sb += 1.0 / tb
                b_e = boltz ** expo if expo != 1.0 else boltz
                ratio = b_e * (pb * sb) / (pa * sa)
                if ratio >= 1.0 or _unif(rng, r) < ratio:
                    c0 = perms[r, rows[0]]
                    for l in range(L):
                        ri = rows[l]
                        old = perms[r, ri]
                        newc = perms[r, rows[l + 1]] if l + 1 < L else c0
                        if sampling and track:
                            counts[group[r], ri, old] += (t - n_burn) - start[r, ri]
                        if sampling:
                            start[r, ri] = t - n_burn
                        ov[r] += int(newc == sigma[ri]) - int(old == sigma[ri])
                        perms[r, ri] = newc
                    for l in range(L):
                        ri = rows[l]
                        invs[r, perms[r, ri]] = ri
                    energy[r] += math.log(boltz) / mu
                    if sampling:
                        n_acc[r] += 1
            if sampling:
                ov_sum[r] += ov[r]
                ov_sq_sum[r] += ov[r] * ov[r]
                en_sum[r] += energy[r]
        if tr < n_trace and t % trace_every == 0:
            for r in range(R):
                trace_ov[r, tr] = ov[r]
                trace_en[r, tr] = energy[r]
            tr += 1

    if track and n_sample > 0:
        for r in range(R):
            g = group[r]
            for i in range(n):
                counts[g, i, perms[r, i]] += n_sample - start[r, i]
    return ov_sum, ov_sq_sum, en_sum, n_acc, trace_ov, trace_en, energy


@njit(cache=True, inline="always")
def _draw_col(Q, AL, i, own, nn, e16, rng, r, kmax):
    """Column ~ qt_i( . | != own) with qt = (1-eps) q_i + eps/n, by rejection
    with at most ``kmax`` tries; returns -1 if every try hit ``own``.
    One 64-bit word gives the column (high 32 bits) and the eps-coin (low 16
    bits); the alias coin uses a fresh 53-bit uniform."""
    for _ in range(kmax):
        y = _next(rng, r)
        c = _bounded(y >> _S32, nn, rng, r)
        if (y & np.uint64(0xFFFF)) >= e16:
            if _unif(rng, r) >= AL[i, c, 0]:
                c = np.int64(AL[i, c, 1])
        if c != own:
            return c
    return -1


@njit(cache=True, inline="always")
def _gk(x, kmax):
    """1 + x + ... + x^(kmax-1): mass factor of the capped rejection step."""
    if x >= 1.0:
        return float(kmax)
    return (1.0 - x ** kmax) / (1.0 - x)


@njit(cache=True)
def _walk_chains(Q, AL, mu, eps16, max_len, kmax, perms, sigma, n_burn,
                 n_sample, n_anneal, group, counts, n_trace, rng):
    """R chains (lock-step) with *ejection-chain* moves of arbitrary length.

    Move.  Pick a row r_0 uniformly.  Row r_l draws a new column
    c_l ~ qt_{r_l}( . | c != pi(r_l)); if c_l = pi(r_0) the walk closes,
    otherwise the displaced row r_{l+1} = pi^{-1}(c_l) continues.  A walk that
    revisits a row or exceeds ``max_len`` rows is a self-loop.  A closed walk
    is the cyclic rotation r_l -> pi(r_{l+1}) of length L <= max_len.
    Every cyclic start generates the same rotation with the same probability,
    so with g the capped-rejection factor:
        fwd/rev = prod_l qt(new_l) g(qt(old_l)) / (qt(old_l) g(qt(new_l)))
    and since exp(mu Delta) = prod q(new)/q(old) (row normalisers and the
    column prices cancel around a cycle), the MH ratio is
        prod_l  [q/qt](new_l) / [q/qt](old_l) * g(qt(new_l)) / g(qt(old_l)),
    which is identically 1 for eps = 0, kmax = inf: the move is then
    rejection-free.  For L = 2 this is a swap.
    """
    R, n = perms.shape
    nn = np.uint64(n)
    eps = eps16 / 65536.0
    e16 = np.uint64(eps16)
    unif_q = eps / n
    i_mass = 1.0 - eps
    rows = np.empty(max_len, dtype=np.int64)

    invs = np.empty((R, n), dtype=np.int64)
    for r in range(R):
        for i in range(n):
            invs[r, perms[r, i]] = i
    mark = np.full(n, -1, dtype=np.int64)     # walk-id stamps (shared, sequential)
    wid = 0
    track = counts.shape[1] == n
    start = np.zeros((R, n), dtype=np.int64)
    ov = np.zeros(R, dtype=np.int64)
    energy = np.zeros(R, dtype=np.float64)
    for r in range(R):
        ov[r] = _overlap(perms[r], sigma)
        for i in range(n):
            energy[r] += math.log(Q[i, perms[r, i]]) / mu

    total = n_burn + n_sample
    trace_ov = np.zeros((R, n_trace), dtype=np.int64)
    trace_en = np.zeros((R, n_trace), dtype=np.float64)
    trace_every = max(total // max(n_trace, 1), 1)
    tr = 0
    ov_sum = np.zeros(R, dtype=np.int64)
    ov_sq_sum = np.zeros(R, dtype=np.float64)
    en_sum = np.zeros(R, dtype=np.float64)
    n_acc = np.zeros(R, dtype=np.int64)
    len_sum = np.zeros(R, dtype=np.int64)

    for t in range(total):
        sampling = t >= n_burn
        for r in range(R):
            expo = 1.0
            if t < n_anneal[r]:
                expo = math.sqrt((t + 1.0) / n_anneal[r])
            wid += 1
            x = _next(rng, r)
            r0 = _bounded(x >> _S32, nn, rng, r)
            c_close = perms[r, r0]
            rows[0] = r0
            mark[r0] = wid
            L = 0
            cur = r0
            for l in range(max_len):
                c = _draw_col(Q, AL, cur, perms[r, cur], nn, e16, rng, r, kmax)
                if c < 0:
                    break
                if c == c_close:
                    L = l + 1
                    break
                nxt = invs[r, c]
                if mark[nxt] == wid or l + 1 >= max_len:
                    break
                mark[nxt] = wid
                rows[l + 1] = nxt
                cur = nxt
            if L >= 2:
                ratio = 1.0
                boltz = 1.0
                for l in range(L):
                    ri = rows[l]
                    rn = rows[l + 1] if l + 1 < L else r0
                    qn = Q[ri, perms[r, rn]]
                    qo = Q[ri, perms[r, ri]]
                    tn = i_mass * qn + unif_q
                    to = i_mass * qo + unif_q
                    boltz *= qn / qo
                    ratio *= (qn * to) / (qo * tn) * _gk(tn, kmax) / _gk(to, kmax)
                if expo != 1.0:
                    ratio *= boltz ** (expo - 1.0)
                if ratio >= 1.0 or _unif(rng, r) < ratio:
                    for l in range(L):
                        ri = rows[l]
                        old = perms[r, ri]
                        newc = perms[r, rows[l + 1]] if l + 1 < L else c_close
                        if sampling:
                            if track:
                                counts[group[r], ri, old] += (t - n_burn) - start[r, ri]
                            start[r, ri] = t - n_burn
                        ov[r] += int(newc == sigma[ri]) - int(old == sigma[ri])
                        perms[r, ri] = newc
                    for l in range(L):
                        ri = rows[l]
                        invs[r, perms[r, ri]] = ri
                    energy[r] += math.log(boltz) / mu
                    if sampling:
                        n_acc[r] += 1
                        len_sum[r] += L
            if sampling:
                ov_sum[r] += ov[r]
                ov_sq_sum[r] += ov[r] * ov[r]
                en_sum[r] += energy[r]
        if tr < n_trace and t % trace_every == 0:
            for r in range(R):
                trace_ov[r, tr] = ov[r]
                trace_en[r, tr] = energy[r]
            tr += 1

    if track and n_sample > 0:
        for r in range(R):
            g = group[r]
            for i in range(n):
                counts[g, i, perms[r, i]] += n_sample - start[r, i]
    return ov_sum, ov_sq_sum, en_sum, n_acc, trace_ov, trace_en, energy, len_sum


@njit(cache=True)
def _uniform_chain(W, mu, perm, sigma, n_burn, n_sample, n_anneal, counts,
                   n_trace):
    n = perm.shape[0]
    inv = np.empty(n, dtype=np.int64)
    for i in range(n):
        inv[perm[i]] = i
    track = counts.shape[0] == n
    start = np.zeros(n, dtype=np.int64)
    ov = _overlap(perm, sigma)
    energy = 0.0
    for i in range(n):
        energy += W[i, perm[i]]

    total = n_burn + n_sample
    trace_ov = np.zeros(n_trace, dtype=np.int64)
    trace_en = np.zeros(n_trace, dtype=np.float64)
    trace_every = max(total // max(n_trace, 1), 1)
    tr = 0
    ov_sum = 0
    ov_sq_sum = 0.0
    en_sum = 0.0
    n_acc = 0
    mu_t = mu
    for t in range(total):
        if t < n_anneal:
            mu_t = mu * math.sqrt((t + 1.0) / n_anneal)
        elif t == n_anneal:
            mu_t = mu
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        a = perm[i]
        if a != j:
            k = inv[j]
            delta = (W[i, j] + W[k, a]) - (W[i, a] + W[k, j])
            if delta >= 0.0 or np.random.random() < math.exp(mu_t * delta):
                if t >= n_burn:
                    s = t - n_burn
                    if track:
                        counts[i, a] += s - start[i]
                        counts[k, j] += s - start[k]
                    start[i] = s
                    start[k] = s
                    n_acc += 1
                ov += (int(j == sigma[i]) + int(a == sigma[k])
                       - int(a == sigma[i]) - int(j == sigma[k]))
                energy += delta
                perm[i] = j
                perm[k] = a
                inv[j] = i
                inv[a] = k
        if t >= n_burn:
            ov_sum += ov
            ov_sq_sum += ov * ov
            en_sum += energy
        if tr < n_trace and t % trace_every == 0:
            trace_ov[tr] = ov
            trace_en[tr] = energy
            tr += 1
    if track and n_sample > 0:
        for i in range(n):
            counts[i, perm[i]] += n_sample - start[i]
    return ov_sum, ov_sq_sum, en_sum, n_acc, trace_ov, trace_en, energy


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------
class Proposal:
    """Proposal data for one instance W (shared by all chains run on W).

    kind :
        ``"walk"``     ejection-chain cyclic rotations of any length <= max_len
                       with natural closure (default; see ``_walk_chains``);
        ``"rotation"`` informed rotations of fixed length 2..5 with forced
                       closure (``cycle_probs``; see ``_informed_chains``);
        ``"uniform"``  the plain swap chain of the write-up.
    sinkhorn_iters : number of Sinkhorn sweeps for the column prices b used in
        q_i(j) propto exp(mu W_ij + b_j)  (0 = no prices).  Exactness does not
        depend on b.
    """

    def __init__(self, W, mu, kind="walk", eps=0.05,
                 cycle_probs=(0.4, 0.25, 0.2, 0.15), sinkhorn_iters=30,
                 max_len=64, kmax=32):
        if kind == "informed":
            kind = "rotation"
        self.kind = kind
        self.eps16 = int(round(float(eps) * 65536))
        self.eps = self.eps16 / 65536.0
        self.mu = float(mu)
        self.max_len = int(max_len)
        self.kmax = int(kmax)
        cp = np.asarray(cycle_probs, dtype=np.float64)
        if not (1 <= cp.size <= 4) or np.any(cp < 0) or cp.sum() <= 0:
            raise ValueError("cycle_probs must have 1..4 nonnegative entries")
        self.len_cdf = np.cumsum(cp / cp.sum())
        self.len_cdf[-1] = 1.0 + 1e-12
        if kind in ("walk", "rotation"):
            n = W.shape[0]
            if sinkhorn_iters > 0:
                b = sinkhorn_log(W, self.mu, int(sinkhorn_iters))
            else:
                b = np.zeros(n)
            self.Q, self.AL, logZ = build_tables(W, self.mu, b)
            # sum_i log Q[i,pi(i)] = mu <Pi,W> + sum_j b_j - sum_i log Z_i
            self.offset = float((logZ.sum() - b.sum()) / self.mu)
            self.W = None
        elif kind == "uniform":
            self.W = W
            self.offset = 0.0
        else:
            raise ValueError(kind)


def _stats(n, mu, n_sample, offset, ov_sum, ov_sq, en_sum, n_acc, tr_ov,
           tr_en, en_final, perm, sigma):
    ns = max(int(n_sample), 1)
    mean_ov = ov_sum / ns
    energy = (en_sum / ns + offset) / n
    return dict(
        q_sample=mean_ov / n,
        q_sample_sd=math.sqrt(max(ov_sq / ns - mean_ov ** 2, 0.0)) / n,
        acc_rate=n_acc / ns,
        # Nishimori: E[<Pi_M, W>]/n = mu at stationarity; sqrt(n)*(e - mu) = O(1)
        energy=energy,
        energy_z=math.sqrt(n) * (energy - mu),
        final_energy=(en_final + offset) / n,
        final_overlap=float(np.mean(perm == sigma)),
        trace_ov=tr_ov / n,
        trace_energy=(tr_en + offset) / n,
    )


def run_chains(W, mu, init_perms, sigma, n_burn, n_sample, counts=None,
               group=None, anneal=None, n_trace=200, seed=0,
               proposal: Proposal | None = None):
    """Run R chains (R = len(init_perms)) on one instance.

    counts : optional (G, n, n) integer array; chain r accumulates into
        counts[group[r]] (default: every chain in group 0).
    anneal : per-chain fraction of the burn-in used to ramp C from 0 to C.
    Returns (final perms (R, n), list of per-chain stats dicts).
    """
    if proposal is None:
        proposal = Proposal(W, mu)
    perms = np.ascontiguousarray(np.array(init_perms, dtype=np.int64))
    if perms.ndim == 1:
        perms = perms[None, :]
    R, n = perms.shape
    sigma = np.ascontiguousarray(sigma, dtype=np.int64)
    if counts is None:
        counts = np.zeros((1, 1, 1), dtype=np.int64)
    if counts.ndim == 2:
        counts = counts[None]
    group = np.zeros(R, np.int64) if group is None else np.asarray(group, np.int64)
    anneal = np.zeros(R) if anneal is None else np.broadcast_to(anneal, (R,))
    n_anneal = np.array([int(a * n_burn) for a in anneal], dtype=np.int64)
    if not isinstance(seed, np.random.SeedSequence):
        seed = np.random.SeedSequence(int(seed))
    seed_numba(int(seed.generate_state(1)[0]))
    if proposal.kind in ("walk", "rotation"):
        if proposal.kind == "walk":
            out = _walk_chains(proposal.Q, proposal.AL, float(mu),
                               proposal.eps16, proposal.max_len, proposal.kmax,
                               perms, sigma, int(n_burn), int(n_sample),
                               n_anneal, group, counts, int(n_trace),
                               make_rng_state(seed, R))
            lens = out[7]
        else:
            out = _informed_chains(proposal.Q, proposal.AL, float(mu),
                                   proposal.eps16, proposal.len_cdf, perms,
                                   sigma, int(n_burn), int(n_sample), n_anneal,
                                   group, counts, int(n_trace),
                                   make_rng_state(seed, R))
            lens = None
        ov_sum, ov_sq, en_sum, n_acc, tr_ov, tr_en, en_fin = out[:7]
        stats = [_stats(n, mu, n_sample, proposal.offset, ov_sum[r], ov_sq[r],
                        en_sum[r], n_acc[r], tr_ov[r], tr_en[r], en_fin[r],
                        perms[r], sigma) for r in range(R)]
        if lens is not None:
            for r in range(R):
                stats[r]["mean_move_len"] = lens[r] / max(n_acc[r], 1)
    else:
        stats = []
        for r in range(R):
            c = counts[group[r]] if counts.shape[1] == n else counts[0]
            out = _uniform_chain(proposal.W, float(mu), perms[r], sigma,
                                 int(n_burn), int(n_sample), int(n_anneal[r]),
                                 c, int(n_trace))
            ov_sum, ov_sq, en_sum, n_acc, tr_ov, tr_en, en_fin = out
            stats.append(_stats(n, mu, n_sample, 0.0, ov_sum, ov_sq, en_sum,
                                n_acc, tr_ov, tr_en, en_fin, perms[r], sigma))
    return perms, stats


def swap_chain(W, mu, init_perm, sigma, n_burn, n_sample, counts=None,
               n_trace=200, seed=0, proposal: Proposal | None = None,
               anneal: float = 0.0):
    """Single-chain convenience wrapper (default proposal: uniform, as in the
    write-up). Returns (final_perm, stats)."""
    if proposal is None:
        proposal = Proposal(W, mu, "uniform")
    perms, stats = run_chains(W, mu, [init_perm], sigma, n_burn, n_sample,
                              counts=counts, anneal=anneal, n_trace=n_trace,
                              seed=seed, proposal=proposal)
    return perms[0], stats[0]
