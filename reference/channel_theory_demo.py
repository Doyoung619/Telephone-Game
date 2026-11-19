"""Exact synthetic-channel example; every number is computed, not LLM data."""
import json
import math
import numpy as np
from schedulers import all_orders, transfer_branch_bound, transfer_score


def run():
    a=np.array([[.95,.70],[0.,.20]])
    b=np.array([[.20,0.],[.70,.60]])
    matrices=np.stack([a,b,np.eye(2),np.eye(2)])
    orders=all_orders(4); initial=np.array([1.,0.])
    products=[]
    for order in orders:
        product=np.eye(2)
        for role in order: product=matrices[role]@product
        products.append(product)
    global_upper=max(float(p.sum(axis=0).max()) for p in products)
    selected=transfer_branch_bound(matrices,initial)
    product=products[orders.index(selected.order)]
    lower=float(product.sum(axis=0).min()); upper=float(product.sum(axis=0).max())
    values=[]; state=initial.copy()
    for cycle in range(1,17):
        state=product@state; observed=float(state.sum())
        assert lower**cycle-1e-12<=observed<=global_upper**cycle+1e-12
        values.append({'cycle':cycle,'exact_survival':observed,
                       'chosen_lower':lower**cycle,'all_orders_upper':global_upper**cycle})
    return {'type':'synthetic_channel_exact_calculation_NOT_LLM_experiment',
            'matrices':matrices.tolist(),'initial':initial.tolist(),
            'chosen_order_indices':selected.order,'lower_per_cycle':lower,
            'global_upper_per_cycle':global_upper,
            'half_life_cycle_lower':math.ceil(math.log(.5)/math.log(lower)),
            'half_life_cycle_upper':math.ceil(math.log(.5)/math.log(upper)),
            'values':values}


if __name__=='__main__': print(json.dumps(run(),indent=2))
