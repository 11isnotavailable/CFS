"""The seven from-scratch architectures and shared, frozen training protocol."""
from functools import partial
import hashlib
import json
from pathlib import Path

SPECS = {
    'V1': (192, 12, 3), 'V2': (256, 12, 4), 'V3': (320, 12, 5),
    'V4': (256, 9, 4), 'V5': (256, 18, 4), 'V6': (256, 12, 2), 'V7': (256, 12, 8),
}
AXES = {'width': ['V1','V2','V3'], 'depth': ['V4','V2','V5'], 'heads': ['V6','V2','V7']}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PROTOCOL = dict(
    epochs=120, seed=42, global_batch=1024, optimizer='adamw', learning_rate=0.001,
    betas=[0.9,0.999], eps=1e-8, weight_decay=0.05, warmup_epochs=5,
    warmup_lr=1e-6, min_lr=1e-5, schedule='cosine', schedule_epochs=120, clip_grad=None,
    drop_path_rate=0.1, dropout=0.0, smoothing=0.1, mixup_alpha=0.8,
    cutmix_alpha=1.0, mixup_prob=1.0, mixup_switch_prob=0.5, mixup_mode='batch',
    auto_augment='rand-m9-mstd0.5-inc1', color_jitter=0.4, re_prob=0.25,
    re_mode='pixel', re_count=1, train_interpolation='bicubic',
    random_crop_scale=[0.08,1.0], random_crop_ratio=[0.75,4/3], hflip=0.5,
    repeated_augmentation=3, mean=MEAN, std=STD, crop_pct=0.875,
    precision='bfloat16', ema_decay=0.99996, checkpoint_selection='epoch120_online',
    early_stopping=None, checkpoint_epochs=[40,60,80,100,120],
    light_cfs=dict(epochs=[60,80,100,120],samples=64,alpha_steps=31,
                   selection='evenly spaced64 offsets in the original fixed512 probe',bootstrap=2000),
    preprocessing=dict(train_short_edge=256,max_long_edge=1024,jpeg_quality=95,jpeg_subsampling=2,
                       validation='pixel-exact original deterministic224 crop cached as CHW uint8'),
    common_budget_checkpoints='40/60/80/100/120 for every model; primary is120 regardless of CFS trends',
    pretrained=False, distillation=False, primary_cfs_samples=512,
    training_overlap_rule='Before training, exclude every training image whose exact SHA256 appears in the full validation set; same index for all models',
    alpha_min=0.0, alpha_max=3.0, alpha_steps=31, valid_source_threshold=1e-5,
    coverage_threshold=0.5, bootstrap=2000, fixed_k_draws=500,
    extension=None,
)


def config(model_id):
    w,d,h = SPECS[model_id]
    return dict(img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                global_pool='token', embed_dim=w, depth=d, num_heads=h, mlp_ratio=4.,
                qkv_bias=True, proj_bias=True, class_token=True, no_embed_class=False,
                reg_tokens=0, pre_norm=False, qk_norm=False, init_values=None,
                drop_rate=0., pos_drop_rate=0., proj_drop_rate=0., attn_drop_rate=0.,
                drop_path_rate=0.1)


def make_model(model_id):
    import torch
    from timm.models.vision_transformer import VisionTransformer
    # Explicit construction, no hub/pretrained/checkpoint loading path.
    return VisionTransformer(**config(model_id), norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))


def flop_counts(model_id):
    """Dense matmul FLOPs, MAC=2; excludes elementwise/norm/softmax/activation costs."""
    w,d,h = SPECS[model_id]; t=197
    attention = d*(8*t*w*w + 4*t*t*w)
    mlp = d*16*t*w*w
    patch = 2*196*w*3*16*16
    classifier = 2*w*1000
    return dict(forward_matmul_flops=attention+mlp+patch+classifier,
                attention_matmul_flops=attention, mlp_matmul_flops=mlp,
                patch_classifier_matmul_flops=patch+classifier)


def controls(model_id):
    h = SPECS[model_id][2]
    return [n for n in (1,2,3) if n < h], [k for k in (2,3) if k <= h]


def protocol_hash():
    return hashlib.sha256(json.dumps(dict(protocol=PROTOCOL, specs=SPECS),sort_keys=True).encode()).hexdigest()


def write_configs(directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    shared=dict(protocol=PROTOCOL, axes=AXES, sha256=protocol_hash())
    (directory/'shared.json').write_text(json.dumps(shared,indent=2))
    for name in SPECS:
        (directory/f'{name}.json').write_text(json.dumps(dict(model_id=name,model_config=config(name),
            norm_eps=1e-6, shared_protocol_sha256=protocol_hash()),indent=2))
