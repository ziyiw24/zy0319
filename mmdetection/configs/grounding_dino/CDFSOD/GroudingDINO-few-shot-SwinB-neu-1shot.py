_base_ = ['GroudingDINO-few-shot-SwinB-neu.py']
# 1-shot：仅覆盖训练标注
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
        ann_file='annotations/1_shot.json',
        data_prefix=dict(img='train/')))

work_dir = '/root/autodl-tmp/ETS/mmdetection_work_dirs/neu_1shot_pgta4'
