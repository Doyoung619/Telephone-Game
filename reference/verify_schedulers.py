"""Mathematical implementation checks on synthetic matrices; NOT LLM results.
Run: python code/verify_schedulers.py
"""
from itertools import permutations
import json
import math
import numpy as np
from schedulers import (all_orders, fit_pairwise, pairwise_dp, fit_adjacent,
    adjacent_dp, transfer_score, transfer_beam, transfer_branch_bound,
    erm, robust_order, contextual_safe, contextual_predict,
    finite_bank_certificate, certify_surrogate)


def verify():
    tests=0; rng=np.random.default_rng(314159)
    for n in [2,3,4,5]:
        orders=all_orders(n)
        for repeat in range(4):
            w=rng.normal(size=(n,n)); w=w-w.T
            v=rng.normal(size=(n,n)); start=rng.normal(size=n); end=rng.normal(size=n)
            pair_values=[sum(w[a,b] for i,a in enumerate(p) for b in p[i+1:]) for p in orders]
            adj_values=[start[p[0]]+end[p[-1]]+sum(v[a,b] for a,b in zip(p,p[1:])) for p in orders]
            assert np.isclose(pairwise_dp(w).value,max(pair_values)); tests+=1
            assert np.isclose(adjacent_dp(start,end,v).value,max(adj_values)); tests+=1
            k=3; matrices=rng.uniform(.01,1,size=(n,k,k))
            matrices/=matrices.sum(axis=1)[:,None,:]
            matrices*=rng.uniform(.25,.99,size=(n,k))[:,None,:]
            initial=rng.dirichlet(np.ones(k)); reward=rng.uniform(.1,1,size=k)
            exact=max(transfer_score(matrices,initial,p,reward) for p in orders)
            for width in [1,3,len(orders)]:
                result=transfer_beam(matrices,initial,reward,width)
                assert result.value <= exact+1e-10 <= result.upper+1e-9
                assert len(set(result.order))==n; tests+=1
            for budget in [0,3,100000]:
                result=transfer_branch_bound(matrices,initial,reward,budget)
                assert result.value <= exact+1e-10 <= result.upper+1e-9
                if budget==100000: assert np.isclose(result.value,exact) and result.gap<1e-9
                tests+=1
    orders=all_orders(4); bank=rng.uniform(0,1,size=(40,len(orders)))
    a1=erm(bank,orders); assert a1.value==bank.mean(axis=0).max(); tests+=1
    intercept,w=fit_pairwise(bank,orders)
    a2=pairwise_dp(w,intercept)
    pred=np.array([intercept+sum(w[a,b] for i,a in enumerate(p) for b in p[i+1:]) for p in orders])
    assert np.isclose(a2.value,pred.max()); tests+=1
    b,s,e,v=fit_adjacent(bank,orders); a3=adjacent_dp(s,e,v,b)
    assert np.isclose(a3.value,max(b+s[p[0]]+e[p[-1]]+sum(v[a,b] for a,b in zip(p,p[1:])) for p in orders)); tests+=1
    cert=certify_surrogate(bank,orders,pred,a2.order)
    assert cert['lower']<=cert['upper'] and cert['error']>=cert['residual']; tests+=1
    a6=robust_order({'domain_a':bank[:20],'domain_b':bank[20:]},orders)
    min_means=np.minimum(bank[:20].mean(0),bank[20:].mean(0))
    assert a6['index']==int(np.argmax(min_means)); tests+=1
    a7=contextual_safe(bank[:20],np.arange(20)*10,bank[20:],np.arange(20)*10,orders)
    assert not a7['activated']; assert len(contextual_predict(a7,[80,180]))==2; tests+=1
    for order in orders:
        c=finite_bank_certificate(bank,orders,order)
        assert 0<=c['lower']<=c['upper']<=1 and c['includes_all_orders']; tests+=1
    print(json.dumps({'verification':'passed','checks':tests,
                      'scope':'synthetic DP/search/certificate checks only; no LLM benchmark run'},ensure_ascii=False))


if __name__=='__main__': verify()
