_base_ = ['GroudingDINO-few-shot-SwinB-uodd.py']
# 1-shot：仅覆盖训练标注
data_root = '../data/UODD/'
metainfo = dict(
    classes=('seacucumber', 'seaurchin', 'scallop'),
    dataset='UODD')
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/1_shot.json',
        data_prefix=dict(img='train/')))

# 1-shot 结果保存到 jieguo20316
work_dir = '/root/autodl-tmp/ETS/jieguo20318/uodd_1shot'
