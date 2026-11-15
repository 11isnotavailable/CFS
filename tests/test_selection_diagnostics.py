import numpy as np
import pytest
from src.selection_diagnostics import ranking_metrics,union_oracle,mask_codes


def test_perfect_and_reversed_ranking_have_expected_bounds():
    y=np.array([[.006,.008,.05,.1,.2,.3]])
    good,_=ranking_metrics(-y,y);bad,_=ranking_metrics(y,y)
    assert good['regret'][0]==0 and good['pair_accuracy'][0]==1 and good['spearman'][0]==1
    assert bad['chosen_true_rank'][0]==6 and bad['pair_accuracy'][0]==0
    assert bad['far_regret'][0]==pytest.approx(.294)
    assert bad['top5_regret'][0]==pytest.approx(.002)
    assert bad['top10_regret'][0]==0


def test_local_swap_global_accuracy_and_shortlist_repair():
    y=np.array([[.006,.008,.05,.1,.2,.3]])
    u=-np.array([[.009,.007,.049,.1,.2,.3]])
    r,_=ranking_metrics(u,y)
    assert r['regret'][0]==pytest.approx(.002)
    assert r['pair_accuracy'][0]==pytest.approx(14/15)
    assert r['near_regret'][0]==r['regret'][0]
    assert r['top3_regret'][0]==0
    np.testing.assert_allclose(union_oracle([u,-y],y,1),[.006])


def test_ties_are_not_invented_ordering_information():
    u=np.zeros((1,6));y=np.array([[.02,.01,.01,.03,.04,.05]])
    r,order=ranking_metrics(u,y)
    assert r['chosen'][0]==0 and r['max_score_ties'][0]==6
    assert r['pair_accuracy'][0]==.5 and np.isnan(r['spearman'][0])
    assert r['chosen_true_rank'][0]==3
    assert order.tolist()==[[0,1,2,3,4,5]]
    with pytest.raises(ValueError):ranking_metrics(u,np.full_like(u,np.nan))


def test_mask_encoding_is_unique_and_budget_preserving():
    masks=np.array([[1,0,1],[0,1,1],[1,1,0]],dtype=bool)
    assert mask_codes(masks).tolist()==[5,6,3]
