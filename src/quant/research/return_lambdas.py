"""Experimental financial metric lambdas, NOT standard NDCG LambdaLoss."""
from __future__ import annotations
import numpy as np
from quant.research.head_lambdaloss import HeadLambdaObjective


class ReturnHeadObjective(HeadLambdaObjective):
    """Swap delta in equal-slot TopK clipped net return.

    Unlike nonnegative relevance grades, losses of different magnitudes remain
    distinguishable. No within-head order penalty: equal portfolio weights make
    a swap inside the head irrelevant to this particular financial metric.
    """
    def __init__(self,query,net,k=20):
        net=np.asarray(net,float)
        if not np.isfinite(net).all():raise ValueError('Only mature known returns may train')
        super().__init__(query,np.clip(net,-.2,.2)+.2,'ranknet_head',k=k)

    def weighted_pairs(self,pred):
        order=np.lexsort((np.arange(self.n),-pred,self.group))
        a,b=order[self.a],order[self.b]
        ga,gb=self.gain[a],self.gain[b]
        boundary=np.abs((self.ra<=self.k).astype(float)-(self.rb<=self.k).astype(float))
        w=self.mc*np.abs(ga-gb)*boundary/self.k
        totals=np.bincount(self.pgroup,weights=w,minlength=len(self.counts))
        w=w/np.maximum(totals[self.pgroup],1e-12)*self.n/max(1,np.count_nonzero(totals))
        keep=w>0
        return a[keep],b[keep],(ga[keep]>gb[keep]).astype(float),w[keep]
