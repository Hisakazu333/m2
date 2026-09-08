"""Old-label certification companion, separate from the finite Bayes model.

Finite Bernoulli evaluator reconstructed from the uploaded PDF. Coefficients use
standard-library arithmetic. Count-lattice evaluation additionally uses NumPy.
No claim of general multiple-repair optimality is made.
"""
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import ceil, exp, floor, inf, lgamma, log, sqrt


def bernoulli_kl(p, q):
    return p*log(p/q)+(1-p)*log((1-p)/(1-q))


@dataclass(frozen=True)
class RepairExperiment:
    pre: tuple = (.5,.9,.6)
    post: tuple = (.8,.2,.2)
    labels: tuple = (0,0,1)
    costs: tuple = (1.,1.,1.)
    repair_success: float = .8
    repair_cost: float = .3

    def __post_init__(self):
        k=len(self.pre)
        if k<2 or any(len(a)!=k for a in (self.post,self.labels,self.costs)):
            raise ValueError("Inconsistent model dimensions")
        if any(not 0<x<1 for x in self.pre+self.post):
            raise ValueError("Full-support Bernoulli channels required")
        if len(set(self.pre))!=k or set(self.labels)!={0,1}:
            raise ValueError("Distinct pre laws and both labels required")
        if any(c<=0 for c in self.costs) or self.repair_cost<0 or not 0<self.repair_success<=1:
            raise ValueError("Invalid costs or repair success")

    @property
    def class_means(self):
        return tuple(dict.fromkeys(self.post))

    @property
    def class_indices(self):
        return tuple(self.class_means.index(q) for q in self.post)

    @property
    def decoders(self):
        return tuple(product((0,1),repeat=len(self.class_means)))

    def bad(self,decoder):
        return tuple(i for i,lab in enumerate(self.labels) if decoder[self.class_indices[i]]!=lab)

    def coefficients(self):
        result=[]
        for i in range(len(self.pre)):
            best=(-1,None)
            for decoder in self.decoders:
                bad=self.bad(decoder)
                if i in bad:
                    continue
                d=min((bernoulli_kl(self.pre[i],self.pre[j]) for j in bad),default=inf)
                if d>best[0]:
                    best=d,decoder
            sep=min(bernoulli_kl(self.pre[i],self.pre[j])
                    for j in range(len(self.pre)) if self.labels[j]!=self.labels[i])
            result.append({"environment":i+1,"rate":best[0],"decoder":best[1],
                           "joint_coefficient":self.costs[i]/best[0],
                           "separated_coefficient":self.costs[i]/sep})
        return result


def _lower_binomial(n,p,k):
    """Lower tail for k < np, evaluated from its largest term."""
    if k<0:
        return 0.
    if k>=n:
        return 1.
    pmf=exp(lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1)+k*log(p)+(n-k)*log(1-p))
    total,term=pmf,pmf
    for j in range(k,0,-1):
        term*=j/(n-j+1)*(1-p)/p
        total+=term
        if term==0 or term<total*1e-17:
            break
    return min(1.,total)


def binomial_cdf(n,p,k):
    if k<0:
        return 0.
    if k>=n:
        return 1.
    if k<n*p:
        return _lower_binomial(n,p,k)
    return max(0.,1-_lower_binomial(n,1-p,n-k-1))


def _class_range(means,index,n):
    order=sorted(range(len(means)),key=lambda j:means[j])
    pos=order.index(index)
    lo,hi=0,n
    current=Fraction(str(means[index]))
    if pos>0:
        other=order[pos-1]
        boundary=n*(current+Fraction(str(means[other])))/2
        lo=ceil(boundary) if index<other else floor(boundary)+1
    if pos<len(order)-1:
        other=order[pos+1]
        boundary=n*(current+Fraction(str(means[other])))/2
        hi=floor(boundary) if index<other else ceil(boundary)-1
    return lo,hi


def _class_probability(n,p,lo,hi):
    if lo>hi:
        return 0.
    if hi<n*p:
        return max(0.,binomial_cdf(n,p,hi)-binomial_cdf(n,p,lo-1))
    if lo>n*p:
        return max(0.,binomial_cdf(n,1-p,n-lo)-binomial_cdf(n,1-p,n-hi-1))
    return max(0.,1-binomial_cdf(n,p,lo-1)-binomial_cdf(n,1-p,n-hi-1))


def finite_protocol(experiment=None, delta=.05, horizon=2400, pre_limit=2000,
                    repair_cap=24, separated=False, selection="factorized"):
    """Evaluate one common rule without supplying a true candidate ID to it.

    Post classes use the nearest-mean classifier. This coincides with maximum
    likelihood for the uploaded paper's symmetric two-class Bernoulli example.
    """
    import numpy as np
    x=experiment or RepairExperiment()
    if not 0<delta<1 or not 0<=pre_limit or repair_cap<1 or pre_limit+repair_cap>=horizon:
        raise ValueError("Leave positive post-repair budget")
    k=len(x.pre)
    beta=log(4*k*(k-1)/delta)
    from .decoder_structure import eligible_codes, decode_code
    if selection not in ("factorized", "enumerated"):
        raise ValueError("Unknown decoder selection implementation")
    mclasses=len(x.class_means)
    ds=(tuple(a for a in x.decoders if not separated or len(set(a))==1)
        if selection=="enumerated" else ())
    bads=[x.bad(a) for a in ds]
    costs=np.array(x.costs)
    labels=np.array(x.labels)
    q=x.repair_success
    failed=(1-q)**repair_cap
    attempts=(1-failed)/q
    alive=np.ones((k,1))
    totals_cost=np.zeros(k)
    totals_n=np.zeros(k)
    totals_error=np.zeros(k)
    stopped=np.zeros(k)
    timeout=np.zeros(k)
    logs=np.log(np.array(x.pre))
    log0=np.log(1-np.array(x.pre))
    post_cache={}

    def after_success_error(n,decoder,i):
        key=(n,decoder,i)
        if key in post_cache:
            return post_cache[key]
        result=0.
        for attempt in range(1,repair_cap+1):
            prob=q*(1-q)**(attempt-1)
            if prob==0:
                continue
            npost=horizon-n-attempt
            err=0.
            for c in range(len(x.class_means)):
                if decoder[c]!=x.labels[i]:
                    lo,hi=_class_range(x.class_means,c,npost)
                    err+=_class_probability(npost,x.post[i],lo,hi)
            result+=prob*err
        post_cache[key]=result
        return result

    for n in range(pre_limit+1):
        counts=np.arange(n+1)
        scores=counts[:,None]*logs+(n-counts[:,None])*log0
        if selection=="factorized":
            selected=eligible_codes(scores,beta,x.class_indices,x.labels,separated)
        else:
            selected=np.full(n+1,-1,dtype=int)
            for decoder,bad in zip(ds,bads):
                eligible=np.zeros(n+1,dtype=bool)
                if not bad:
                    eligible[:]=True
                else:
                    for witness in range(k):
                        if witness not in bad:
                            eligible |= np.all(scores[:,witness,None]-scores[:,bad]>=beta,axis=1)
                code=sum(v*(1 << (mclasses-1-c)) for c,v in enumerate(decoder))
                selected[(selected<0)&eligible]=code
        mle=labels[np.argmax(scores,axis=1)]
        for code in np.unique(selected[selected>=0]):
            decoder=decode_code(code,mclasses)
            mask=selected==code
            if not np.any(mask):
                continue
            mass=alive[:,mask].sum(axis=1)
            if not np.any(mass):
                continue
            stopped+=mass
            totals_n+=mass*n
            totals_cost+=mass*(costs*n+x.repair_cost*attempts+failed*costs*(horizon-n-repair_cap))
            for i in range(k):
                if separated:
                    failure_error=mass[i]*int(decoder[0]!=x.labels[i])
                else:
                    failure_error=alive[i,mask & (mle!=x.labels[i])].sum()
                totals_error[i]+=failed*failure_error+mass[i]*after_success_error(n,decoder,i)
            alive[:,mask]=0
        if n==pre_limit:
            timeout=alive.sum(axis=1)
            totals_cost+=timeout*costs*horizon
            totals_n+=timeout*n
            for i in range(k):
                totals_error[i]+=alive[i,mle!=x.labels[i]].sum()
            break
        nxt=np.zeros((k,n+2))
        for i,p in enumerate(x.pre):
            nxt[i,:-1]+=alive[i]*(1-p)
            nxt[i,1:]+=alive[i]*p
        alive=nxt
    means=sorted(x.class_means)
    delta_post=min((b-a for a,b in zip(means,means[1:])),default=inf)
    post_bound=0. if len(means)==1 else 2*exp(-(horizon-pre_limit-repair_cap)*delta_post**2/2)
    return [{"policy":"separated" if separated else "joint", "environment":i+1,
             "cost":float(totals_cost[i]),"pre_samples":float(totals_n[i]),
             "error":float(totals_error[i]),"timeout":float(timeout[i]),
             "conservative_error_bound":min(1.,delta/4+float(timeout[i])+failed+post_bound),
             "mass_residual":float(stopped[i]+timeout[i]-1),"threshold":beta,
             "numerics":"double precision count lattice; stable binomial tails",
             "selection":selection}
            for i in range(k)]
