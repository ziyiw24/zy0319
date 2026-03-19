_base_ = ['GroudingDINO-few-shot-SwinB.py']

# dataset: UODD, 3 classes, 10-shot（必须显式覆盖 dataloader.dataset，否则 base 已展开的 NEU-DET 不会替换）
data_root = '../data/UODD/'
metainfo = dict(
    classes=('seacucumber', 'seaurchin', 'scallop'),
    dataset='UODD')

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
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[8, 12], gamma=0.1)
]
train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=5)
default_hooks = dict(
    checkpoint=dict(by_epoch=True, interval=5, max_keep_ckpts=1))

# Prompt 配置：B 贴近 UODD 语义（去掉不贴合的颜色/场景属性，避免负向 prompt 干扰）
model = dict(
    prompt_template='a photo of a {class}',
    prompt_template_map=dict(
        UODD='an underwater photo of a {class}',
    ),
    prompt_attrs=dict(),
    negative_prompt_template=None,
    # Fix BERT tokenization: "seacucumber"/"seaurchin" are compound words that
    # BERT splits into poor sub-word tokens; space-separated forms are much better.
    class_name_corrections=dict(
        seacucumber='sea cucumber',
        seaurchin='sea urchin',
    ),
)

work_dir = '/root/autodl-tmp/ETS/jieguo20317/uodd_10shot'
