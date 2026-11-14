import itertools
import json
import numpy as np
import pytest
from scripts.analyze_matched_scaling import matched, subsets, fixed_metrics, estimate
from scripts.analyze_qwen_cfs import expected_best_two, minimum_cover
from scripts.analyze_functional_capacity import spectral_effective_rank


def test_exact_expectations_enumerated():
    a=np.array([[-2.,-.5,0.,0.,.7,1.], [.3,.3,.3,.3,.3,.3]])
    for k in range(1,7):
        expected=[np.mean([max(row[list(c)]) for c in itertools.combinations(range(6),k)]) for row in a]
        np.testing.assert_allclose(matched(a,k),expected,atol=1e-14)
    np.testing.assert_allclose(matched(a,2),[expected_best_two(r) for r in a])
    with pytest.raises(ValueError): matched(a,7)


def test_subsets_boundary_and_reproducibility():
    np.testing.assert_array_equal(subsets(6,6,500,np.random.default_rng(42)),[np.arange(6)])
    a=subsets(12,4,500,np.random.default_rng(42))
    np.testing.assert_array_equal(a,subsets(12,4,500,np.random.default_rng(42)))
    assert all(len(set(r))==4 for r in a)
    assert a.min()>=0 and a.max()<12
    with pytest.raises(ValueError): subsets(6,7,500,np.random.default_rng(42))


def test_induced_graph_matches_original_with_invalid_heads():
    rng=np.random.default_rng(7)
    s=rng.uniform(-.5,1,(6,6)); np.fill_diagonal(s,np.nan)
    v=np.array([True,False,True,True,False,True])
    draws=subsets(6,4,10,rng)
    result=fixed_metrics(s,v,draws)
    for row,idx in zip(result,draws):
        sub=s[np.ix_(idx,idx)]; mask=v[idx]
        assert row[0]==minimum_cover(sub,mask,.5)/4
        assert row[1]==pytest.approx(spectral_effective_rank(sub[None],mask[None])[0]/4)
    full=fixed_metrics(s,v,subsets(6,6,500,rng))[0]
    assert full[0]==minimum_cover(s,v,.5)/6
    assert full[1]==pytest.approx(spectral_effective_rank(s[None],v[None])[0]/6)


def test_source_weighted_cluster_estimator():
    num=np.array([[1.,4.],[9.,2.]])
    den=np.array([[2.,4.],[10.,4.]])
    assert estimate(num,den,np.array([[0,1]]))[0]==pytest.approx((10/12+6/8)/2)
    # Same input sampled twice retains both of its layers together.
    assert estimate(num,den,np.array([[0,0]]))[0]==pytest.approx((1/2+4/4)/2)


def test_end_to_end_pairing_and_reports(tmp_path, monkeypatch):
    import pandas as pd
    from scripts.analyze_matched_scaling import analyze
    from scripts.summarize_matched_scaling import main as summarize
    from scripts.report_matched_scaling import main as report
    from scripts.validate_matched_scaling import validate
    source=tmp_path/'synthetic'; source.mkdir()
    (source/'metadata.json').write_text(json.dumps(dict(model='test-only',heads=6,selected_layers=[11],dataset_size=50000)))
    rng=np.random.default_rng(13)
    drops=np.ones((4,1,6)); drops[0,0,:3]=0
    substitution=rng.uniform(0,1,(4,1,6,6))
    np.savez(source/'shard_00000.npz',single_drop=drops,substitution=substitution,dataset_index=[30,10,40,20])
    analysis=tmp_path/'analysis'; analysis.mkdir()
    for i in range(2):
        entry=dict(name=f'test_{i}',family='ViT',model_id='test-only',parameters=(i+1)*1000000,heads=6,samples=4,layers=[11],N=[2,4,5],K=[4,6],path=str(source),input_identity_sha256='synthetic-shared-inputs')
        out=analysis/entry['name']; out.mkdir()
        analyze(entry,tmp_path,out,repeats=20,bootstrap=40)
        validate(out)
        prov=json.loads((out/'provenance.json').read_text())
        assert prov['ordered_input_ids']==[10,20,30,40]
    figures=tmp_path/'figures'
    monkeypatch.setattr('sys.argv',['summarize','--input',str(analysis),'--output',str(figures)])
    summarize()
    contrasts=pd.read_csv(figures/'paired_contrasts.csv')
    exact=contrasts[contrasts.metric.isin(['raw','matched_2','cover_full','rank_full'])]
    np.testing.assert_allclose(exact[['delta','ci_low','ci_high']],0,atol=1e-12)
    monkeypatch.setattr('sys.argv',['report','--figures',str(figures),'--analysis',str(analysis),'--historical',str(tmp_path/'no-history')])
    report()
    assert (figures/'EXPERIMENT_REPORT.md').exists()
    assert (figures/'figure1_matched_N.pdf').stat().st_size>1000
