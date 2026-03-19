_base_ = ['GroudingDINO-few-shot-SwinB.py']

# dataset: clipart1k, 20 classes VOC, 10-shot（必须显式覆盖 dataloader.dataset，否则 base 已展开的 NEU-DET 不会替换）
data_root = '../data/clipart1k/'
metainfo = dict(
    classes=(
        'sheep', 'chair', 'boat', 'bottle', 'diningtable', 'sofa', 'cow',
        'motorbike', 'car', 'aeroplane', 'cat', 'train', 'person', 'bicycle',
        'pottedplant', 'bird', 'dog', 'bus', 'tvmonitor', 'horse'))

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/10_shot.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/')))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/')))

# evaluator 使用当前 data_root
val_evaluator = dict(ann_file=data_root + 'annotations/test.json')
test_evaluator = dict(ann_file=data_root + 'annotations/test.json')

# 15 epoch，只保留 epoch_15
max_epochs = 15
param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[5, 12], gamma=0.1)
]
train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=5)
default_hooks = dict(
    checkpoint=dict(by_epoch=True, interval=5, max_keep_ckpts=1))
