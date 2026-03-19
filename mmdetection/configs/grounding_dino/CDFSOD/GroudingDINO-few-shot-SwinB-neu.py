_base_ = ['GroudingDINO-few-shot-SwinB.py']

# dataset: NEU-DET, 6 defect classes, 10-shot（须显式覆盖 dataloader.dataset）
data_root = '../data/NEU-DET/'
metainfo = dict(
    classes=(
        'crazing',
        'inclusion',
        'patches',
        'pitted_surface',
        'rolled-in_scale',
        'scratches',
    ))

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

# 新目录，不覆盖 pgta3 旧权重（PGTA 通用改进：小框 blend + 自适应 EMA）
work_dir = '/root/autodl-tmp/ETS/mmdetection_work_dirs/neu_10shot_pgta4'
