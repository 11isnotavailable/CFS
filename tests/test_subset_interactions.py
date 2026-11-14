import numpy as np
from src.subset_interactions import interaction_masks,quartet_interactions,additive_crossfit


def test_quartets_are_actual_masks_with_fixed_background():
    masks,pairs,quartets,random_ids=interaction_masks(42)
    for index,keep in enumerate((3,6,9)):
        assert np.all(masks[random_ids[index]].sum(1)==keep)
        assert len(np.unique(random_ids[index]))==48
        for (i,j),indices in zip(pairs,quartets[index]):
            t,ti,tj,tij=masks[indices]
            assert t.sum()==keep-2 and not t[i] and not t[j]
            assert np.flatnonzero(ti^t).tolist()==[i]
            assert np.flatnonzero(tj^t).tolist()==[j]
            assert np.array_equal(tij,ti|tj)


def test_true_additivity_and_known_quadratic_interaction():
    masks,pairs,q,_=interaction_masks(17)
    linear=masks@np.arange(12)+7
    np.testing.assert_allclose(quartet_interactions(linear,q),0)
    quadratic=linear+2*masks[:,0]*masks[:,1]
    measured=quartet_interactions(quadratic,q)
    np.testing.assert_allclose(measured[:,0],2)
    np.testing.assert_allclose(measured[:,1:],0)


def test_crossfit_linear_reconstruction_and_duplicate_grouping():
    masks,_,_,ids=interaction_masks(23);x=masks[ids[1]];y=x@np.arange(12)+3
    stat,pred,fold=additive_crossfit(x,y)
    assert stat['rmse']<1e-10 and stat['rank_min']==12
    xd=np.concatenate([x,x[:2]]);yd=np.concatenate([y,y[:2]])
    _,_,fd=additive_crossfit(xd,yd)
    np.testing.assert_array_equal(fd[:2],fd[-2:])
