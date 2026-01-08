import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .code_dataset import RealCodeDataset, FakeCodeDataset
from .datasets import dataset_folder


# def get_dataset(opt):
#     dset_lst = []
#     for cls in opt.classes:
#         root = opt.dataroot + '/' + cls
#         #dset = dataset_folder(opt, root)
#         dset = FakeCodeDataset(
#             root=root,
#             split=opt.train_split if opt.isTrain else opt.val_split,
#             opt=opt
#         )
#         dset_lst.append(dset)
#     return torch.utils.data.ConcatDataset(dset_lst)


def get_dataset(opt):
    split = opt.phase  # train / val

    if opt.fake_code:
        return FakeCodeDataset(opt.dataroot, split=split)
    else:
        # ✅ 添加 max_samples 参数
        max_samples = getattr(opt, 'max_samples', None)  # 从opt读取，默认None（全部使用）

        return RealCodeDataset(
            dataroot=opt.dataroot,
            split=split,
            max_length=getattr(opt, 'max_length', 512),  # 默认512
            max_samples=max_samples  # ✅ 传入参数
        )


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader
