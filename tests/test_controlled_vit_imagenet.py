"""Checks specific to the new controlled experiment, including H=2 boundaries."""
import copy
import itertools
import numpy as np
import torch
from src.controlled_vit_imagenet import SPECS, make_model, controls, PROTOCOL, config, flop_counts
from scripts.analyze_matched_scaling import matched, fixed_metrics, subsets
from timm.scheduler import CosineLRScheduler


def test_fixed_capacity_identical_initial_parameters():
    torch.set_num_threads(2)
    states=[]; counts=[]
    for name in ['V2','V6','V7']:
        torch.manual_seed(42); model=make_model(name)
        states.append({k:v.clone() for k,v in model.state_dict().items()})
        counts.append(sum(p.numel() for p in model.parameters()))
    assert counts==[9982184]*3
    for key in states[0]:
        assert torch.equal(states[0][key],states[1][key])
        assert torch.equal(states[0][key],states[2][key])
    assert flop_counts('V2')==flop_counts('V6')==flop_counts('V7')


def test_architectural_axes():
    assert [SPECS[n][0]//SPECS[n][2] for n in ['V1','V2','V3']]==[64]*3
    assert [(SPECS[n][0],SPECS[n][2]) for n in ['V4','V2','V5']]==[(256,4)]*3
    assert PROTOCOL['epochs']==120 and not PROTOCOL['pretrained']
    assert all(config(n)['num_classes']==1000 for n in SPECS)


def test_h2_controls_no_fabricated_matched2():
    assert controls('V6')==([1],[2])
    values=np.array([[.25],[.8]])
    np.testing.assert_array_equal(matched(values,1),values[:,0])
    draws=subsets(2,2,500,np.random.default_rng(42))
    assert draws.shape==(1,2)
    s=np.array([[np.nan,.25],[.8,np.nan]])
    full=fixed_metrics(s,np.array([True,True]),draws)
    assert full.shape==(1,2) and np.isfinite(full).all()


def test_exact_matched_for_width_heads():
    for h in [3,4,5]:
        v=np.linspace(-.2,.9,h-1)[None]
        for n in [1,2]:
            want=np.mean([max(x) for x in itertools.combinations(v[0],n)])
            np.testing.assert_allclose(matched(v,n),[want],atol=1e-14)


def test_optimizer_scheduler_rng_roundtrip():
    torch.manual_seed(42)
    model=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Dropout(.1),torch.nn.Linear(8,2))
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    sched=CosineLRScheduler(opt,t_initial=20,warmup_t=2,t_in_epochs=False)
    x=torch.randn(3,4)
    def step(m,o,s,i):
        o.zero_grad(); loss=m(x).square().mean(); loss.backward(); o.step(); s.step_update(i)
    step(model,opt,sched,1)
    saved=copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=torch.get_rng_state()))
    step(model,opt,sched,2)
    resumed=copy.deepcopy(model); resumed.load_state_dict(saved['model'])
    o=torch.optim.AdamW(resumed.parameters(),lr=.001); s=CosineLRScheduler(o,t_initial=20,warmup_t=2,t_in_epochs=False)
    o.load_state_dict(saved['optimizer']); s.load_state_dict(saved['scheduler']); torch.set_rng_state(saved['rng'])
    step(resumed,o,s,2)
    for a,b in zip(model.parameters(),resumed.parameters()): torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_packed_images_preserve_pixels_labels_and_order(tmp_path):
    import io
    import json
    from PIL import Image
    from src.indexed_image_dataset import IndexedImages
    from scripts.pack_controlled_imagenet import DTYPE
    raw=[]
    for color in [(10,20,30),(90,80,70)]:
        stream=io.BytesIO(); Image.new('RGB',(7,9),color).save(stream,format='PNG'); raw.append(stream.getvalue())
    (tmp_path/'val_0000.images').write_bytes(b''.join(raw))
    (tmp_path/'classes.json').write_text(json.dumps({'a':0,'b':1}))
    index=np.array([(0,b'a/x.png',0,0,len(raw[0])),(1,b'b/y.png',0,len(raw[0]),len(raw[1]))],dtype=DTYPE)
    np.save(tmp_path/'val_index.npy',index)
    data=IndexedImages(tmp_path,'val')
    for i in range(2):
        image,label=data[i]; expected=Image.open(io.BytesIO(raw[i])).convert('RGB')
        assert label==i; np.testing.assert_array_equal(np.array(image),np.array(expected))
        # In-place modification of one returned view cannot poison decode reuse.
        image.putpixel((0,0),(0,0,0))
        repeated,_=data[i]; np.testing.assert_array_equal(np.array(repeated),np.array(expected))


def test_common_budget_and_fixed_light_probe():
    from scripts.collect_predictor_dataset import evenly_subsample_indices
    assert PROTOCOL['epochs']==PROTOCOL['schedule_epochs']==120
    assert PROTOCOL['early_stopping'] is None
    assert PROTOCOL['checkpoint_epochs']==[40,60,80,100,120]
    assert PROTOCOL['light_cfs']['epochs']==[60,80,100,120]
    full=np.linspace(0,49999,512,dtype=int)
    small=evenly_subsample_indices(full,64)
    assert len(np.unique(small))==64 and set(small)<=set(full)
    assert small[0]==full[0] and small[-1]==full[-1]


def test_validation_pixel_cache_matches_original_transform(tmp_path):
    import json
    from PIL import Image
    from timm.data import create_transform
    from src.controlled_vit_imagenet import MEAN,STD
    from src.indexed_image_dataset import IndexedImages
    from scripts.pack_controlled_imagenet import DTYPE
    rng=np.random.default_rng(42); image=Image.fromarray(rng.integers(0,256,(315,477,3),dtype=np.uint8))
    kwargs=dict(input_size=(3,224,224),is_training=False,mean=MEAN,std=STD,interpolation='bicubic',crop_pct=.875)
    pixels=np.asarray(create_transform(**kwargs,use_prefetcher=True)(image))
    expected=create_transform(**kwargs,use_prefetcher=False)(image)
    (tmp_path/'val_0000.images').write_bytes(pixels.tobytes())
    (tmp_path/'classes.json').write_text(json.dumps({'a':0})); (tmp_path/'storage.json').write_text(json.dumps({'val':'chw_uint8'}))
    np.save(tmp_path/'val_index.npy',np.array([(0,b'a/image',0,0,pixels.nbytes)],dtype=DTYPE))
    cached,_=IndexedImages(tmp_path,'val')[0]
    normalized=(torch.from_numpy(cached).float()/255-torch.tensor(MEAN)[:,None,None])/torch.tensor(STD)[:,None,None]
    torch.testing.assert_close(normalized,expected,rtol=0,atol=1e-6)


def test_stability_uses_original_unclipped_scores_and_input_order(tmp_path):
    import json
    from scripts.analyze_controlled_vit_stability import matrices
    (tmp_path/'metadata.json').write_text(json.dumps(dict(model='controlled_V6',heads=2,
        selected_layers=[11],sample_indices=[7,2],dataset_size=50000)))
    drop=np.array([[[.1,1e-6]],[[.2,.3]]])
    sub=np.array([[[[0,.2],[1e-6,0]]],[[[0,.1],[.6,0]]]])
    np.savez(tmp_path/'shard_00000.npz',single_drop=drop,substitution=sub,dataset_index=[7,2])
    ids,scores,valid=matrices(tmp_path)
    assert ids.tolist()==[2,7]
    np.testing.assert_allclose(scores[:,0,1],[.5,-1])
    assert valid.tolist()==[[True,True],[True,False]]
    assert np.isnan(scores[:,np.arange(2),np.arange(2)]).all()
