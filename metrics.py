import torch
from torch import Tensor
from numpy import ndarray

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _check_input(*args: Tensor | ndarray):
    return tuple([torch.from_numpy(arg, dtype=torch.int16, device=_device) if isinstance(arg, ndarray) else arg for arg in args])


def _check_continuous(*args: Tensor):
    res = []
    for arg in args:
        if arg.unique().numel() != arg.max() + 1:
            res.append(remap_label_(arg))
        else:
            res.append(arg)
    return tuple(res)


def all_in_one(true: Tensor | ndarray, pred: Tensor | ndarray, match_iou: float = 0.5):
    true, pred = _check_input(true, pred)
    true, pred = _check_continuous(true, pred)
    return _all_in_one_impl(true, pred, match_iou)


def aji(true: Tensor | ndarray, pred: Tensor | ndarray):
    true, pred = _check_input(true, pred)
    true, pred = _check_continuous(true, pred)
    return _aji_impl(true, pred)


def pq(true: Tensor | ndarray, pred: Tensor | ndarray, match_iou: float = 0.5):
    true, pred = _check_input(true, pred)
    true, pred = _check_continuous(true, pred)
    return _pq_impl(true, pred, match_iou)


def dice2(true: Tensor | ndarray, pred: Tensor | ndarray, match):
    true, pred = _check_input(true, pred)
    true, pred = _check_continuous(true, pred)
    return _dice2_impl(true, pred)


def mpq(true_inst: Tensor | ndarray, true_cls, pred_inst: Tensor | ndarray, pred_cls: Tensor | ndarray):
    true_inst, pred_inst, true_cls, pred_cls = _check_input(true_inst, pred_inst, true_cls, pred_cls)
    return _mpq_impl(true_inst, true_cls, pred_inst, pred_cls)


def _all_in_one_impl(true: Tensor, pred: Tensor, match_iou: float = 0.5):
    # 计算PQ、DQ、SQ、AJI、DICE2,DICE，这里假设true和pred的值都是从1开始的连续整数，0为背景，即经过remap_label函数处理过的

    # 使用int16类型可以减少内存，使用cuda可以加速运算
    true, pred = true.to(torch.int16).to(_device), pred.to(torch.int16).to(_device)
    # 计算DICE
    true_bool = true != 0
    pred_bool = pred != 0
    dice = 2 * (true_bool & pred_bool).sum().item() / (true_bool.sum().item() + pred_bool.sum().item() + 1e-6)
    del true_bool, pred_bool
    # 这里必须使用CPU版本的bincount，使用cuda会非常慢，并且要先reshape再cpu，否则也会变慢
    true_counts = torch.bincount(true.reshape(-1).cpu())
    pred_counts = torch.bincount(pred.reshape(-1).cpu())
    # tp 表示true和pred的组合，counts表示组合出现的次数
    tp = torch.stack([true, pred], dim=2)  # H * W * 2
    tp = tp[(true != 0) & (pred != 0)]  # 去掉背景
    tp, counts = tp.unique(dim=0, return_counts=True)  # 统计组合出现的次数，该次数即为交集的大小
    # 分别计算每个组合的交集和并集，虽然可以直接存储交并比，但是AJI需要用到交集和并集，所以这里分开存储
    pairwise_inter = torch.zeros((len(true_counts) - 1, len(pred_counts) - 1), dtype=torch.float32)
    pairwise_union = torch.zeros((len(true_counts) - 1, len(pred_counts) - 1), dtype=torch.float32)
    # 用于计算DICE2
    dice2_overall_total, dice2_overall_inter = 0, 0

    for (true_id, pred_id), count in zip(tp, counts):
        total = true_counts[true_id] + pred_counts[pred_id] - count  # 并集
        pairwise_inter[true_id - 1, pred_id - 1] = count  # 交集
        pairwise_union[true_id - 1, pred_id - 1] = total
        dice2_overall_total += total + count
        dice2_overall_inter += count
    # 计算IOU
    pairwise_iou = pairwise_inter / (pairwise_union + 1e-6)
    # DICE2
    dice2 = (2 * dice2_overall_inter / (dice2_overall_total + 1e-6)).item()
    # PQ
    paired_iou = pairwise_iou[pairwise_iou > match_iou]  # 交并比大于match_iou的为匹配的,默认0.5，也可以改为更大的值，match_iou小于0.5会出错
    tp_ = len(paired_iou)
    fp = len(pred_counts) - tp_ - 1
    fn = len(true_counts) - tp_ - 1
    dq = tp_ / (tp_ + 0.5 * fp + 0.5 * fn + 1e-6)
    sq = paired_iou.sum().item() / (tp_ + 1e-6)
    # AJI
    pairwise_iou = pairwise_iou.max(1)  # 每个true 实例对应的最大交并比
    paired_pred = pairwise_iou.indices  # 每个true 实例对应的最大交并比的pred 实例
    pairwise_iou = pairwise_iou.values  # 每个true 实例对应的最大交并比的值

    paired_true = torch.nonzero(pairwise_iou > 0.0).squeeze(1)  # 交并比大于0的为匹配的，就是有匹配的pred的true实例的索引-1
    unpaired_true = torch.nonzero(pairwise_iou == 0.0).squeeze(1) + 1  # 交并比等于0的为不匹配的，就是没有匹配的pred的true实例的索引
    paired_pred = paired_pred[paired_true]  # 有匹配的pred的true实例对应的pred实例的索引-1

    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum().item()  # 匹配实例对的交集
    overall_union = (pairwise_union[paired_true, paired_pred]).sum().item()  # 匹配实例对的并集

    paired_pred = (paired_pred + 1).tolist()  # 匹配实例对 的pred实例的索引

    unpaired_pred = torch.tensor([idx for idx in range(1, len(pred_counts)) if idx not in paired_pred], dtype=torch.int)  # 没有匹配的pred实例的索引

    overall_union += true_counts[unpaired_true].sum().item()  # 没有匹配的pred实例面积
    overall_union += pred_counts[unpaired_pred].sum().item()  # 没有匹配的true实例面积

    aji_score = overall_inter / overall_union  # AJI

    return {'PQ': dq * sq, 'DQ': dq, 'SQ': sq, 'AJI': aji_score, 'DICE2': dice2, 'DICE': dice}


# 对AJI、PQ、DICE2进行优化，提升运算速度并减少内存占用，均优化10倍以上


def _aji_impl(true: Tensor, pred: Tensor):
    true, pred = true.to(torch.int16).to(_device), pred.to(torch.int16).to(_device)
    true_id_list = torch.unique(true)[1:]
    pred_id_list = torch.unique(pred)[1:].tolist()
    true_counts = torch.bincount(true.reshape(-1).cpu())
    pred_counts = torch.bincount(pred.reshape(-1).cpu())
    tp = torch.stack([true, pred], dim=2)
    tp = tp[(true != 0) & (pred != 0)]
    tp, counts = tp.unique(dim=0, return_counts=True)
    pairwise_inter = torch.zeros((len(true_id_list), len(pred_id_list)), dtype=torch.float32)
    pairwise_union = torch.zeros((len(true_id_list), len(pred_id_list)), dtype=torch.float32)
    for (true_id, pred_id), count in zip(tp, counts):
        total = true_counts[true_id] + pred_counts[pred_id] - count
        pairwise_inter[true_id - 1, pred_id - 1] = count
        pairwise_union[true_id - 1, pred_id - 1] = total

    pairwise_iou = pairwise_inter / (pairwise_union + 1e-6)
    pairwise_iou = pairwise_iou.max(1)
    paired_pred = pairwise_iou.indices
    pairwise_iou = pairwise_iou.values

    paired_true = torch.nonzero(pairwise_iou > 0.0).squeeze(1)
    unpaired_true = torch.nonzero(pairwise_iou == 0.0).squeeze(1) + 1
    paired_pred = paired_pred[paired_true]

    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum().item()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum().item()

    paired_pred = (paired_pred + 1).tolist()

    unpaired_pred = torch.tensor([idx for idx in pred_id_list if idx not in paired_pred], dtype=torch.int)

    overall_union += true_counts[unpaired_true].sum().item()
    overall_union += pred_counts[unpaired_pred].sum().item()

    aji_score = overall_inter / overall_union
    return aji_score


def _pq_impl(true: Tensor, pred: Tensor, match_iou=0.5):
    assert match_iou >= 0.0, "Cant' be negative"
    true, pred = true.to(torch.int16).to(_device), pred.to(torch.int16).to(_device)

    true_counts = torch.bincount(true.reshape(-1).cpu())
    pred_counts = torch.bincount(pred.reshape(-1).cpu())
    tp = torch.stack([true, pred], dim=2)
    tp = tp[(true != 0) & (pred != 0)]
    tp, counts = tp.unique(dim=0, return_counts=True)
    pairwise_iou = torch.zeros((len(true_counts) - 1, len(pred_counts) - 1), dtype=torch.float32)

    for (true_id, pred_id), count in zip(tp, counts):
        total = true_counts[true_id] + pred_counts[pred_id] - count
        pairwise_iou[true_id - 1, pred_id - 1] = count / (total + 1e-6)

    paired_iou = pairwise_iou[pairwise_iou > match_iou]

    tp = len(paired_iou)
    fp = len(pred_counts) - tp - 1
    fn = len(true_counts) - tp - 1

    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
    sq = paired_iou.sum().item() / (tp + 1e-6)

    return dq, sq, dq * sq


def _dice2_impl(true: Tensor, pred: Tensor):
    true, pred = true.to(torch.int16).to(_device), pred.to(torch.int16).to(_device)
    true_counts = torch.bincount(true.reshape(-1).cpu())
    pred_counts = torch.bincount(pred.reshape(-1).cpu())
    tp = torch.stack([true, pred], dim=2)
    tp = tp[(true != 0) & (pred != 0)]
    tp, counts = tp.unique(dim=0, return_counts=True)
    overall_total, overall_inter = 0, 0
    for (true_id, pred_id), count in zip(tp, counts):
        overall_total += true_counts[true_id] + pred_counts[pred_id]
        overall_inter += count
    return 2 * overall_inter / (overall_total + 1e-6)


# 计算mPQ
def _mpq_impl(true_inst: Tensor, true_cls: Tensor, pred_inst: Tensor, pred_cls: Tensor):

    true_cls_id_list = torch.unique(true_cls)[1:].tolist()
    mpq = []
    for cls in true_cls_id_list:
        true_inst_c = true_inst.clone()
        true_inst_c[true_cls != cls] = 0
        true_inst_c = remap_label_(true_inst_c)

        pred_inst_c = pred_inst.clone()
        pred_inst_c[pred_cls != cls] = 0
        pred_inst_c = remap_label_(pred_inst_c)

        _, _, pq = _pq_impl(true_inst_c, pred_inst_c)
        mpq.append(pq)
    return sum(mpq) / len(mpq)


def remap_label(pred: Tensor):
    ids = torch.unique(pred)[1:].tolist()
    new_pred = torch.zeros_like(pred)
    for i, id in enumerate(ids):
        new_pred[pred == id] = i + 1
    return new_pred


# remap_label 原地版
def remap_label_(pred: Tensor):
    ids = torch.unique(pred)[1:].tolist()
    for i, id in enumerate(ids):
        pred[pred == id] = i + 1
    return pred
