"""Original CFS collector with the frozen from-scratch DeiT preprocessing.

No changes to interventions, KL, alpha search, validity or saved shard schema.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import collect_predictor_dataset as original
from src.controlled_vit_imagenet import MEAN, STD, PROTOCOL

load_original=original.create_timm_model


def load_with_frozen_transform(name, checkpoint):
    if checkpoint is None:
        raise ValueError('A controlled, trained checkpoint is mandatory')
    model,mapping=load_original(name,checkpoint)
    model.pretrained_cfg=dict(input_size=(3,224,224),mean=MEAN,std=STD,
        interpolation='bicubic',crop_pct=PROTOCOL['crop_pct'],crop_mode='center',num_classes=1000)
    return model,mapping


if __name__=='__main__':
    original.create_timm_model=load_with_frozen_transform
    original.main()
