"""Measured mask quartets and identifiable held-out additive diagnostics."""
import itertools
import numpy as np


def interaction_masks(seed,heads=12,budgets=(3,6,9)):
    rng=np.random.default_rng(seed);pairs=np.asarray(list(itertools.combinations(range(heads),2)))
    masks=[];lookup={}
    def index(indices):
        key=tuple(sorted(indices))
        if key not in lookup:
            lookup[key]=len(masks);mask=np.zeros(heads,dtype=bool);mask[list(key)]=True;masks.append(mask)
        return lookup[key]
    quartets=[];random_ids=[]
    for keep in budgets:
        local=[]
        for i,j in pairs:
            candidates=[h for h in range(heads) if h not in (i,j)]
            background=rng.choice(candidates,keep-2,replace=False).tolist()
            local.append([index(background),index(background+[int(i)]),index(background+[int(j)]),index(background+[int(i),int(j)])])
        quartets.append(local)
        combinations=list(itertools.combinations(range(heads),keep))
        chosen=rng.choice(len(combinations),48,replace=False)
        random_ids.append([index(combinations[i]) for i in chosen])
    return np.asarray(masks),pairs,np.asarray(quartets),np.asarray(random_ids)


def quartet_interactions(kl,indices):
    values=np.asarray(kl)[indices]
    return values[...,3]-values[...,1]-values[...,2]+values[...,0]


def additive_crossfit(masks,values,seed=42):
    """4-fold mask-group CV, minimum-norm fixed-budget regression with intercept."""
    masks=np.asarray(masks,float);values=np.asarray(values,float)
    unique,inverse=np.unique(masks,axis=0,return_inverse=True)
    # Duplicate masks stay in one fold; average repeated numerical observations.
    y=np.array([values[inverse==i].mean() for i in range(len(unique))])
    rng=np.random.default_rng(seed);order=rng.permutation(len(unique))
    folds=np.empty(len(unique),int);folds[order]=np.arange(len(unique))%4
    pred=np.zeros(len(unique));constant=np.zeros(len(unique));train_errors=[];ranks=[]
    for fold in range(4):
        train=folds!=fold;test=~train;xm=unique[train].mean(0);ym=y[train].mean()
        centered=unique[train]-xm
        w=np.linalg.pinv(centered,rcond=1e-10)@(y[train]-ym)
        pred[test]=(unique[test]-xm)@w+ym;constant[test]=ym
        train_errors.extend(((unique[train]-xm)@w+ym-y[train])**2)
        ranks.append(int(np.linalg.matrix_rank(centered))+1)
    errors=pred-y
    return dict(rmse=float(np.sqrt(np.mean(errors**2))),mae=float(abs(errors).mean()),
        train_rmse=float(np.sqrt(np.mean(train_errors))),constant_rmse=float(np.sqrt(np.mean((constant-y)**2))),
        residual_sd=float(errors.std(ddof=1)),rank_min=min(ranks),rank_max=max(ranks),unique_masks=len(unique)),pred[inverse],folds[inverse]
