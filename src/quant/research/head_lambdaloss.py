"""Head-focused metric lambdas for boosted trees (research only).

LambdaRank: exact swap delta of NDCG@k.
NDCG-Loss2: Wang et al. (2018), equations 15 and 18.
NDCG-Loss2++: equation 19, full-rank rho + mu * delta, head truncation.
Ranks/weights are frozen during each boosting update. Log base is natural;
query normalization absorbs its constant difference from the paper's log2.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.special import expit


def fit_grade_cuts(net):
    positive=np.asarray(net,dtype=float)
    positive=positive[np.isfinite(positive)&(positive>0)]
    if not len(positive):return np.array([.005,.015,.03])
    return np.quantile(positive,[.5,.8,.95])


def gains(net,cuts):
    """Negative/zero/unknown net return has zero gain; positives get grades 1..4."""
    x=np.asarray(net,dtype=float)
    grade=np.where(np.isfinite(x)&(x>0),1+np.searchsorted(cuts,x,side='right'),0)
    return np.exp2(grade)-1


def metric_weights(ra,rb,ga,gb,idcg,method,k,mu=5.):
    ra,rb=np.asarray(ra),np.asarray(rb)
    head=(ra<=k)|(rb<=k)
    discount_a=1/np.log2(ra+1);discount_b=1/np.log2(rb+1)
    if method=='ranknet_head':return head.astype(float)
    rho=np.abs(discount_a-discount_b)
    gap=np.abs(ra-rb)
    delta=1/np.log2(np.maximum(gap,1)+1)-1/np.log2(np.maximum(gap,1)+2)
    delta=np.where(gap>0,delta,0.)
    if method=='lambdarank':
        positional=np.abs(discount_a*(ra<=k)-discount_b*(rb<=k))
    elif method=='lambdaloss2':positional=delta
    elif method=='lambdaloss2pp':positional=rho+mu*delta
    else:raise ValueError(f'Unknown method: {method}')
    return head*positional*np.abs(ga-gb)/np.maximum(idcg,1e-12)


class HeadLambdaObjective:
    """Re-rank each query on EVERY update and compare its current head to tail.

    All head-head pairs are exact. Uniform tail-rank samples per head have
    inverse inclusion weights. Queries receive equal total lambda weight.
    Tail positions are seeded once and map to different documents as ranks move.
    No predicted score used here is used to define chronological OOF groups.
    """
    def __init__(self,query,gain,method,k=20,tail_samples=32,seed=73,dates=None,cross_only=False):
        query=np.asarray(query);self.gain=np.asarray(gain,float);self.method=method;self.k=k
        if k<1 or tail_samples<1 or not np.isfinite(self.gain).all() or (self.gain<0).any():
            raise ValueError('Invalid head loss inputs')
        if len(query)!=len(self.gain) or (query[1:]<query[:-1]).any():
            raise ValueError('Rows must be sorted by query')
        self.query=query;self.n=len(query);self.dates=None if dates is None else np.asarray(dates)
        if cross_only and dates is None:raise ValueError('Cross-date objective requires dates')
        self.cross_only=cross_only
        _,self.group,self.counts=np.unique(query,return_inverse=True,return_counts=True)
        self.starts=np.r_[0,np.cumsum(self.counts)[:-1]]
        rng=np.random.default_rng(seed);a=[];b=[];mc=[];pid=[];idcg=[]
        for g,(start,n) in enumerate(zip(self.starts,self.counts)):
            h=min(k,n);aa,bb=np.triu_indices(h,1)
            ga,gb=[aa],[bb];gw=[np.ones(len(aa))]
            tail=n-h
            if tail:
                take=min(tail_samples,tail)
                for rank in range(h):
                    j=h+rng.choice(tail,take,replace=False)
                    ga.append(np.full(take,rank));gb.append(j);gw.append(np.full(take,tail/take))
            aa=np.concatenate(ga);bb=np.concatenate(gb)
            a.append(aa+start);b.append(bb+start);mc.append(np.concatenate(gw));pid.append(np.full(len(aa),g))
            ideal=np.sort(self.gain[start:start+n])[::-1][:h]
            idcg.append(np.sum(ideal/np.log2(np.arange(h)+2)))
        self.a=np.concatenate(a);self.b=np.concatenate(b);self.mc=np.concatenate(mc);self.pgroup=np.concatenate(pid)
        self.ra=self.a-self.starts[self.pgroup]+1;self.rb=self.b-self.starts[self.pgroup]+1
        self.idcg=np.asarray(idcg)
        self.calls=0;self.last_pairs=0

    def weighted_pairs(self,pred):
        pred=np.asarray(pred)
        if len(pred)!=self.n or not np.isfinite(pred).all():raise ValueError('Invalid predictions')
        order=np.lexsort((np.arange(self.n),-pred,self.group))
        a,b=order[self.a],order[self.b]
        ga,gb=self.gain[a],self.gain[b]
        w=self.mc*metric_weights(self.ra,self.rb,ga,gb,self.idcg[self.pgroup],self.method,self.k)
        w*=ga!=gb
        if self.cross_only:w*=self.dates[a]!=self.dates[b]
        totals=np.bincount(self.pgroup,weights=w,minlength=len(self.counts))
        active=np.count_nonzero(totals)
        w=w/np.maximum(totals[self.pgroup],1e-12)*(self.n/max(active,1))
        keep=w>0
        return a[keep],b[keep],(ga[keep]>gb[keep]).astype(float),w[keep]

    def __call__(self,pred,dm=None):
        a,b,y,w=self.weighted_pairs(pred)
        p=expit(pred[a]-pred[b]);v=w*(p-y);h=w*p*(1-p)
        grad=np.bincount(a,weights=v,minlength=self.n)-np.bincount(b,weights=v,minlength=self.n)
        hess=2*(np.bincount(a,weights=h,minlength=self.n)+np.bincount(b,weights=h,minlength=self.n))
        self.calls+=1;self.last_pairs=len(a)
        return grad,np.maximum(hess,1e-8)


def head_metrics(frame,score,cuts,ks=(1,5,10,20)):
    """Unknown selected returns stay in the head, with zero NDCG gain / stress -10%."""
    d=frame[['symbol','date','outcome']].assign(score=score)
    d=d.sort_values(['date','score','symbol'],ascending=[True,False,True],kind='stable')
    rows=[]
    for date,g in d.groupby('date',sort=False):
        gain=gains(g.outcome,cuts);ideal=np.sort(gain)[::-1]
        for k in ks:
            h=min(k,len(g));disc=1/np.log2(np.arange(h)+2);den=float(ideal[:h]@disc)
            selected=g.iloc[:h].outcome
            rows.append(dict(date=date,k=k,ndcg=float(gain[:h]@disc/den) if den>0 else np.nan,
                             stress_net=selected.fillna(-.1).mean(),win=selected.gt(0).mean(),
                             unknown=int(selected.isna().sum()),picked=h,
                             recall=float(np.sum(gain[:h]>0)/np.sum(gain>0)) if (gain>0).any() else np.nan))
    return pd.DataFrame(rows)
