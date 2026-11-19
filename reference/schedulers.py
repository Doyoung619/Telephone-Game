"""Seven reference schedulers. Pure NumPy; never calls a language model.

Rows of a score bank are independent source RECORDS, columns permutations.
Average decoding replicates within each record before calling these functions.
Guarantees concern the population generating those records, not every input.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
import heapq
import math
import numpy as np


@dataclass
class SearchResult:
    order: tuple[int, ...]
    value: float
    upper: float
    expanded: int = 0

    @property
    def gap(self):
        return max(0.0, self.upper - self.value)


def all_orders(n: int):
    if n < 1:
        raise ValueError('n must be positive')
    return list(permutations(range(n)))


def _bank(scores, orders):
    y = np.asarray(scores, dtype=float)
    if y.ndim != 2 or y.shape[0] == 0 or y.shape[1] != len(orders):
        raise ValueError('scores must have shape [source_records, permutations]')
    if not np.all(np.isfinite(y)) or np.min(y) < 0 or np.max(y) > 1:
        raise ValueError('observed scores must be finite in [0,1]')
    if not orders or len(set(map(tuple, orders))) != len(orders):
        raise ValueError('orders must be distinct')
    n = len(orders[0])
    if any(tuple(sorted(p)) != tuple(range(n)) for p in orders):
        raise ValueError('each order must use each role exactly once')
    return y


def uniform_radius(n_records: int, n_policies: int, delta=0.05):
    if n_records < 1 or n_policies < 1 or not 0 < delta < 1:
        raise ValueError('invalid confidence arguments')
    return math.sqrt(math.log(2 * n_policies / delta) / (2 * n_records))


def finite_bank_certificate(scores, orders, selected_order, delta=0.05):
    """Simultaneous bounds valid even for a bank-selected static order.

    Upper is for the listed class; it covers ALL permutations only if the bank
    contains them all. No guarantee for an X0-dependent policy via this upper.
    """
    y = _bank(scores, orders)
    mu = y.mean(axis=0)
    e = uniform_radius(len(y), len(orders), delta)
    j = list(map(tuple, orders)).index(tuple(selected_order))
    lower = max(0., mu[j] - e)
    return dict(lower=float(lower), upper=float(min(1., mu.max() + e)),
                epsilon=e, mean=float(mu[j]), n_records=len(y),
                includes_all_orders=len(orders) == math.factorial(len(orders[0])),
                fidelity_lower_if_score_is_S=lower / (2. - lower))


def erm(scores, orders):
    """A1: exact empirical maximizer of the provided static policy class."""
    y = _bank(scores, orders)
    mu = y.mean(axis=0)
    j = int(np.argmax(mu))
    return SearchResult(tuple(orders[j]), float(mu[j]), float(mu[j]))


def _ridge(x, y, ridge=1e-3):
    """Fit raw surrogate, no clipping before structured optimization."""
    if ridge < 0:
        raise ValueError('ridge must be nonnegative')
    z = np.column_stack([np.ones(len(x)), x])
    reg = np.eye(z.shape[1]) * ridge
    reg[0, 0] = 0.
    return np.linalg.lstsq(z.T @ z / len(z) + reg,
                           z.T @ y / len(z), rcond=None)[0]


def pair_features(orders, n):
    pairs = [(a,b) for a in range(n) for b in range(a+1,n)]
    rows = []
    for p in orders:
        pos = {a:i for i,a in enumerate(p)}
        rows.append([1. if pos[a] < pos[b] else -1. for a,b in pairs])
    return np.asarray(rows), pairs


def fit_pairwise(scores, orders, ridge=1e-3):
    """A2 surrogate: b + sum_{i<j} W[pi_i, pi_j]."""
    y = _bank(scores, orders)
    n = len(orders[0])
    x, pairs = pair_features(orders, n)
    coef = _ridge(x, y.mean(axis=0), ridge)
    w = np.zeros((n,n))
    for c,(a,b) in zip(coef[1:], pairs):
        w[a,b], w[b,a] = c, -c
    return float(coef[0]), w


def pairwise_dp(w, intercept=0.):
    w = np.asarray(w, dtype=float)
    n = len(w)
    if w.shape != (n,n) or n < 1 or not np.all(np.isfinite(w)):
        raise ValueError('w must be a finite square matrix')
    size = 1 << n
    # incoming[mask,b] = sum_{a in mask} W[a,b]
    incoming = np.zeros((size,n))
    for mask in range(1,size):
        bit = mask & -mask
        incoming[mask] = incoming[mask ^ bit] + w[bit.bit_length()-1]
    dp = np.full(size,-np.inf); dp[0] = 0.
    parent = np.full(size,-1,dtype=int)
    for mask in range(1,size):
        for b in range(n):
            if mask & (1 << b):
                prev = mask ^ (1 << b)
                v = dp[prev] + incoming[prev,b]
                if v > dp[mask]:
                    dp[mask], parent[mask] = v,b
    p=[]; mask=size-1
    while mask:
        b=int(parent[mask]); p.append(b); mask ^= 1 << b
    p.reverse(); v=float(dp[-1]+intercept)
    return SearchResult(tuple(p),v,v,expanded=size-1)


def adjacent_features(orders, n):
    x=np.zeros((len(orders),2*n+n*n))
    for row,p in enumerate(orders):
        x[row,p[0]]=1.; x[row,n+p[-1]]=1.
        for a,b in zip(p,p[1:]): x[row,2*n+a*n+b]=1.
    return x


def fit_adjacent(scores, orders, ridge=1e-3):
    """A3: b + start[first] + sum adjacent V[a,b] + end[last]."""
    y=_bank(scores,orders); n=len(orders[0])
    coef=_ridge(adjacent_features(orders,n),y.mean(axis=0),ridge)
    return float(coef[0]),coef[1:1+n],coef[1+n:1+2*n],coef[1+2*n:].reshape(n,n)


def adjacent_dp(start, end, transition, intercept=0.):
    start=np.asarray(start,dtype=float); end=np.asarray(end,dtype=float)
    v=np.asarray(transition,dtype=float); n=len(start)
    if n < 1 or end.shape!=(n,) or v.shape!=(n,n):
        raise ValueError('incompatible shapes')
    dp=np.full((1<<n,n),-np.inf); parent=np.full((1<<n,n),-1,dtype=int)
    for b in range(n): dp[1<<b,b]=start[b]
    for mask in range(1,1<<n):
        for b in range(n):
            if not mask&(1<<b): continue
            prev=mask^(1<<b)
            if not prev: continue
            for a in range(n):
                if prev&(1<<a):
                    val=dp[prev,a]+v[a,b]
                    if val>dp[mask,b]: dp[mask,b],parent[mask,b]=val,a
    last=int(np.argmax(dp[-1]+end)); val=float(dp[-1,last]+end[last]+intercept)
    p=[]; mask=(1<<n)-1
    while mask:
        p.append(last); prev=int(parent[mask,last]); mask^=1<<last; last=prev
    return SearchResult(tuple(reversed(p)),val,val,expanded=n*(1<<n))


def _transfer(matrices, initial, reward=None):
    p=np.asarray(matrices,dtype=float); x=np.asarray(initial,dtype=float)
    if p.ndim!=3 or p.shape[1]!=p.shape[2] or p.shape[0]<1:
        raise ValueError('matrices must have shape [roles,k,k]')
    n,k,_=p.shape
    r=np.ones(k) if reward is None else np.asarray(reward,dtype=float)
    if x.shape!=(k,) or r.shape!=(k,) or not all(np.all(np.isfinite(t)) for t in [p,x,r]):
        raise ValueError('invalid state/reward')
    if np.min(p)<0 or np.min(x)<0 or x.sum()>1+1e-9 or np.min(r)<0 or np.max(r)>1:
        raise ValueError('nonnegative mass and reward in [0,1] required')
    if np.max(p.sum(axis=1))>1+1e-9:
        raise ValueError('each matrix column must sum to at most one')
    return p,x,r


def transfer_score(matrices, initial, order, reward=None):
    p,x,r=_transfer(matrices,initial,reward)
    for a in order: x=p[a]@x
    return float(r@x)


def relaxed_vectors(matrices, reward):
    """u_R=max_a P_a.T u_(R\a), coordinatewise, an admissible upper bound.

    The relaxation may choose different next actions for different hidden states.
    It is therefore an upper bound, NOT a feasible state-oblivious schedule.
    """
    p=np.asarray(matrices); n=p.shape[0]
    @lru_cache(None)
    def solve(mask):
        if mask==0: return np.asarray(reward)
        return np.max(np.stack([p[a].T@solve(mask^(1<<a))
                                for a in range(n) if mask&(1<<a)]),axis=0)
    solve((1<<n)-1)
    return solve


def transfer_beam(matrices, initial, reward=None, width=4):
    """A4. Global pruned bounds yield a VALID, possibly loose search certificate."""
    p,x,r=_transfer(matrices,initial,reward); n=len(p)
    if width<1: raise ValueError('width must be positive')
    u=relaxed_vectors(p,r); full=(1<<n)-1
    beam=[((),full,x)]; discarded=-np.inf; expanded=0
    for depth in range(n):
        children=[]
        for prefix,mask,state in beam:
            for a in range(n):
                if mask&(1<<a):
                    newmask=mask^(1<<a); newstate=p[a]@state
                    children.append((float(u(newmask)@newstate),prefix+(a,),newmask,newstate))
                    expanded+=1
        children.sort(key=lambda t:(-t[0],t[1]))
        if len(children)>width: discarded=max(discarded,max(t[0] for t in children[width:]))
        beam=[(t[1],t[2],t[3]) for t in children[:width]]
    best=max(beam,key=lambda t:float(r@t[2])); val=float(r@best[2])
    return SearchResult(best[0],val,float(max(val,discarded)),expanded)


def transfer_branch_bound(matrices, initial, reward=None, max_nodes=100000):
    """A5. Anytime upper/lower certificate for the FITTED transfer surrogate."""
    p,x,r=_transfer(matrices,initial,reward); n=len(p)
    if max_nodes<0: raise ValueError('max_nodes must be nonnegative')
    u=relaxed_vectors(p,r); full=(1<<n)-1
    # Obtain a feasible incumbent without any language-model inference.
    prefix=(); mask=full; state=x.copy()
    while mask:
        a=max([a for a in range(n) if mask&(1<<a)],
              key=lambda a:float(u(mask^(1<<a))@(p[a]@state)))
        state=p[a]@state; prefix+=(a,); mask^=1<<a
    best=prefix; val=float(r@state)
    heap=[(-float(u(full)@x),0,(),full,x)]; counter=0; expanded=0
    while heap and expanded<max_nodes:
        neg,_,pref,mask,state=heapq.heappop(heap)
        if -neg<=val+1e-14: continue
        expanded+=1
        for a in range(n):
            if not mask&(1<<a): continue
            rest=mask^(1<<a); nx=p[a]@state; npref=pref+(a,)
            if rest==0:
                score=float(r@nx)
                if score>val: val,best=score,npref
            else:
                upper=float(u(rest)@nx)
                if upper>val+1e-14:
                    counter+=1; heapq.heappush(heap,(-upper,counter,npref,rest,nx))
    upper=max(val,-heap[0][0] if heap else val)
    return SearchResult(best,val,float(upper),expanded)


def robust_order(scenario_banks, orders, delta=.05):
    """A6: maximize min_g LCB(mu_g,pi). Scenarios fixed before inspection."""
    if not scenario_banks: raise ValueError('at least one scenario is required')
    means=[]; radii=[]; names=[]; g=len(scenario_banks); m=len(orders)
    for name,bank in scenario_banks.items():
        y=_bank(bank,orders); means.append(y.mean(axis=0)); names.append(name)
        radii.append(math.sqrt(math.log(2*g*m/delta)/(2*len(y))))
    means=np.asarray(means); radii=np.asarray(radii)
    lower=np.min(means-radii[:,None],axis=0)
    j=int(np.argmax(lower))
    return dict(order=tuple(orders[j]), index=j,
                worst_case_lower=float(max(0.,lower[j])),
                robust_optimum_upper=float(min(1.,np.max(np.min(means+radii[:,None],axis=0)))),
                means={name:float(means[k,j]) for k,name in enumerate(names)},
                radii=dict(zip(names,map(float,radii))))


def contextual_safe(fit_scores, fit_lengths, safety_scores, safety_lengths,
                    orders, threshold=120, minimum_leaf=3, delta=.05):
    """A7: train two fixed length bins, certify ONE frozen policy vs baseline.

    Safety records may NOT have been used for model/threshold selection.
    A tiny safety sample commonly makes the bound vacuous: fall back honestly.
    """
    tr=_bank(fit_scores,orders); sa=_bank(safety_scores,orders)
    lengths=np.asarray(fit_lengths); safe_lengths=np.asarray(safety_lengths)
    if lengths.shape!=(len(tr),) or safe_lengths.shape!=(len(sa),):
        raise ValueError('one word length per record is required')
    base=int(np.argmax(tr.mean(axis=0))); by_leaf=[]
    for is_short in [True,False]:
        mask=(lengths<=threshold)==is_short
        by_leaf.append(int(np.argmax(tr[mask].mean(axis=0))) if mask.sum()>=minimum_leaf else base)
    candidate=np.where(safe_lengths<=threshold,by_leaf[0],by_leaf[1])
    cs=sa[np.arange(len(sa)),candidate]; bs=sa[:,base]
    diff=cs-bs
    gain_radius=math.sqrt(2*math.log(2/delta)/len(sa))
    gain_lcb=float(diff.mean()-gain_radius)
    activate=gain_lcb>0
    chosen=cs if activate else bs
    absolute_radius=math.sqrt(math.log(8/delta)/(2*len(sa)))
    return dict(threshold=threshold, baseline_index=base,
                leaf_indices=by_leaf if activate else [base,base],
                proposed_leaf_indices=by_leaf, activated=bool(activate),
                mean_paired_gain=float(diff.mean()), paired_gain_lcb=gain_lcb,
                selected_score_lower=float(max(0.,chosen.mean()-absolute_radius)),
                n_safety_records=len(sa), orders=[tuple(p) for p in orders])


def contextual_predict(model, lengths):
    return [model['orders'][model['leaf_indices'][0 if length<=model['threshold'] else 1]]
            for length in lengths]


def certify_surrogate(scores, orders, predicted_scores, selected_order,
                       surrogate_upper=None, delta=.05):
    """Uniform all-order empirical residual + finite-record sampling bound.

    Covers only listed orders, and only this aggregate static-order population.
    Using the same bank to fit surrogate is allowed because the sampling event
    is uniform over the fixed finite policy class; residual itself is descriptive.
    """
    y=_bank(scores,orders); mu=y.mean(axis=0)
    pred=np.asarray(predicted_scores,dtype=float)
    if pred.shape!=(len(orders),) or not np.all(np.isfinite(pred)):
        raise ValueError('one finite surrogate score per order')
    j=list(map(tuple,orders)).index(tuple(selected_order))
    residual=float(np.max(np.abs(mu-pred)))
    radius=uniform_radius(len(y),len(orders),delta)
    error=residual+radius
    upper=float(pred.max() if surrogate_upper is None else surrogate_upper)
    if upper+1e-10<pred.max(): raise ValueError('claimed surrogate upper is invalid')
    return dict(residual=residual, sampling_radius=radius, error=error,
                lower=float(max(0.,pred[j]-error)),
                upper=float(min(1.,upper+error)),
                regret_upper=float(min(1.,upper-pred[j]+2*error)))
