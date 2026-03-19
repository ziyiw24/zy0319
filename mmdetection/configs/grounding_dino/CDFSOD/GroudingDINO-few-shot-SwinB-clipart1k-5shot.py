_base_ = ['GroudingDINO-few-shot-SwinB-clipart1k.py']
# 5-shot：仅覆盖训练标注
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
        ann_file='annotations/5_shot.json',
        data_prefix=dict(img='train/')))
