_base_ = ['GroudingDINO-few-shot-SwinB-uodd.py']

# Use local pretrained weight to avoid remote download.
load_from = '../weights/grounding_dino_swin-b_pretrain_all.pth'

data_root = '../data/UODD/'
metainfo = dict(
    classes=('seacucumber', 'seaurchin', 'scallop'),
    dataset='UODD')

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/5_shot.json',
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

model = dict(
    enable_pgta=False,
    prompt_template='{class}',
    prompt_template_map=dict(),
    prompt_attrs=dict(),
    negative_prompt_template=None,
)

work_dir = '/root/autodl-tmp/ETS/tips0316/uodd_5shot_nopgta'
