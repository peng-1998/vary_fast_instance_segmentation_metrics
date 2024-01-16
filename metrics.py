import torch
from torch import Tensor
from numpy import ndarray

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _check_input(*args: Tensor | ndarray):
    return tuple([torch.from_numpy(arg, dtype=torch.int16, device=_device) if isinstance(arg, ndarray) else arg for arg in args])


def _check_continuous(*args: Tensor):
    res = []
    for arg in args:
        max_id = arg.max().item()
        if arg.unique().numel() != max_id + 1:
            if max_id > arg.numel():
                arg = remap_label_fast(arg, max_id)
            else:
                res.append(remap_label_fast(arg))
        else:
            res.append(arg)
    return tuple(res)


def all_in_one(true: Tensor | ndarray, pred: Tensor | ndarray, match_iou: float = 0.5):
    """
    Calculate PQ, DQ, SQ, AJI, DICE2, DICE.
    Args:
        true: Tensor or ndarray, shape is (H, W).
        pred: Tensor or ndarray, shape is (H, W).
        match_iou: float, the threshold of iou for matching instances for PQ.
    Return:
        dict: {'PQ': float, 'DQ': float, 'SQ': float, 'AJI': float, 'DICE2': float, 'DICE': float}
    """
    true, pred = _check_input(true, pred)
    true, pred = _check_continuous(true, pred)
    return _all_in_one_impl(true, pred, match_iou)


def aji(true: Tensor | ndarray, pred: Tensor | ndarray):
    """
    Calculate AJI.
    Args:
        true: Tensor or ndarray, shape is (H, W).
        pred: Tensor or ndarray, shape is (H, W).
    Return:
        float: AJI.
    """
    true, pred = _check_input(true, pred)
    true, pred = _check_continuous(true, pred)
    return _aji_impl(true, pred)


def pq(true: Tensor | ndarray, pred: Tensor | ndarray, match_iou: float = 0.5):
    """
    Calculate PQ, DQ, SQ.
    Args:
        true: Tensor or ndarray, shape is (H, W).
        pred: Tensor or ndarray, shape is (H, W).
        match_iou: float, the threshold of iou for matching instances.
    Return:
        float: DQ.
        float: SQ.
        float: PQ.
    """
    true, pred = _check_input(true, pred)
    true, pred = _check_continuous(true, pred)
    return _pq_impl(true, pred, match_iou)


def dice2(true: Tensor | ndarray, pred: Tensor | ndarray, match):
    """
    Calculate DICE2.
    Args:
        true: Tensor or ndarray, shape is (H, W).
        pred: Tensor or ndarray, shape is (H, W).
    Return: 
        float: DICE2.
    """
    true, pred = _check_input(true, pred)
    true, pred = _check_continuous(true, pred)
    return _dice2_impl(true, pred)


def mpq(true_inst: Tensor | ndarray, true_cls, pred_inst: Tensor | ndarray, pred_cls: Tensor | ndarray):
    """
    Calculate mPQ.
    Args:
        true_inst: Tensor or ndarray, shape is (H, W).
        true_cls: Tensor or ndarray, shape is (H, W).
        pred_inst: Tensor or ndarray, shape is (H, W).
        pred_cls: Tensor or ndarray, shape is (H, W).
    Return:
        float: mPQ.
    """
    true_inst, pred_inst, true_cls, pred_cls = _check_input(true_inst, pred_inst, true_cls, pred_cls)
    return _mpq_impl(true_inst, true_cls, pred_inst, pred_cls)


def _all_in_one_impl(true: Tensor, pred: Tensor, match_iou: float = 0.5):
    # Using cuda can speed up the calculation. Since these values will be used as indexes later, they must be int, long or bool types. Here use int.
    true, pred = true.to(torch.int).to(_device), pred.to(torch.int).to(_device)
    # Calculate DICE
    true_bool, pred_bool = true != 0, pred != 0
    inter = true_bool & pred_bool  # This variable is used later
    dice = 2 * inter.sum().item() / (true_bool.sum().item() + pred_bool.sum().item() + 1e-6)
    del true_bool, pred_bool  # Release memory
    # Statistics the number of pixels of each instance.
    # Here must use the CPU version of bincount, using cuda will be very slow.
    # And reshape first and then cpu, otherwise it will be slow too.
    true_counts = torch.bincount(true.reshape(-1).cpu()).to(_device)
    pred_counts = torch.bincount(pred.reshape(-1).cpu()).to(_device)
    # Statistics intersection of each instance pair.
    tp = torch.stack([true, pred], dim=2)  # H * W * 2
    tp = tp[inter]  # Save the intersection of each instance pair
    tp, counts = tp.unique(dim=0, return_counts=True)  # Statistics the number of pixels of the intersection of each instance pair
    # Calculate the intersection and union of each instance pair.
    # Although the intersection and union can be stored directly, the AJI needs to use the intersection and union, so they are stored separately here.
    pairwise_inter = torch.zeros((len(true_counts) - 1, len(pred_counts) - 1), dtype=torch.long, device=_device)
    pairwise_union = torch.zeros((len(true_counts) - 1, len(pred_counts) - 1), dtype=torch.long, device=_device)
    true_ids, pred_ids = tp[:, 0], tp[:, 1]
    pairwise_inter[true_ids - 1, pred_ids - 1] = counts  # intersection
    pairwise_union[true_ids - 1, pred_ids - 1] = true_counts[true_ids] + pred_counts[pred_ids] - counts  # union
    # calculate DICE2
    dice2_overall_total = (true_counts[true_ids] + pred_counts[pred_ids]).sum().item()
    dice2_overall_inter = counts.sum().item()
    dice2 = 2 * dice2_overall_inter / (dice2_overall_total + 1e-6)
    # Calculate IOU
    pairwise_iou = pairwise_inter / (pairwise_union + 1e-6)
    # PQ
    paired_iou = pairwise_iou[pairwise_iou > match_iou]
    tp_ = len(paired_iou)
    fp = len(pred_counts) - tp_ - 1
    fn = len(true_counts) - tp_ - 1
    dq = tp_ / (tp_ + 0.5 * (fp + fn) + 1e-6)
    sq = paired_iou.sum().item() / (tp_ + 1e-6)
    # AJI
    pairwise_iou = pairwise_iou.max(1)
    paired_pred = pairwise_iou.indices
    pairwise_iou = pairwise_iou.values
    paired_true = torch.nonzero(pairwise_iou > 0.0).squeeze(1)
    unpaired_true = torch.nonzero(pairwise_iou == 0.0).squeeze(1) + 1
    paired_pred = paired_pred[paired_true]
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum().item()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum().item()
    pred_paired = torch.ones(len(pred_counts), dtype=torch.bool)
    pred_paired[paired_pred + 1] = False
    pred_paired[0] = False
    unpaired_pred = torch.nonzero(pred_paired).squeeze(1)
    overall_union += true_counts[unpaired_true].sum().item()
    overall_union += pred_counts[unpaired_pred].sum().item()
    aji_score = overall_inter / overall_union  # AJI

    return {'PQ': dq * sq, 'DQ': dq, 'SQ': sq, 'AJI': aji_score, 'DICE2': dice2, 'DICE': dice}


def _aji_impl(true: Tensor, pred: Tensor):
    true, pred = true.to(torch.int).to(_device), pred.to(torch.int).to(_device)
    true_counts = torch.bincount(true.reshape(-1).cpu()).to(_device)
    pred_counts = torch.bincount(pred.reshape(-1).cpu()).to(_device)
    tp = torch.stack([true, pred], dim=2)
    tp = tp[(true != 0) & (pred != 0)]
    tp, counts = tp.unique(dim=0, return_counts=True)
    pairwise_inter = torch.zeros((len(true_counts) - 1, len(pred_counts) - 1), dtype=torch.long, device=_device)
    pairwise_union = torch.zeros((len(true_counts) - 1, len(pred_counts) - 1), dtype=torch.long, device=_device)
    true_ids, pred_ids = tp[:, 0], tp[:, 1]
    pairwise_inter[true_ids - 1, pred_ids - 1] = counts
    pairwise_union[true_ids - 1, pred_ids - 1] = true_counts[true_ids] + pred_counts[pred_ids] - counts

    pairwise_iou = pairwise_inter / (pairwise_union + 1e-6)
    pairwise_iou = pairwise_iou.max(1)
    paired_pred = pairwise_iou.indices
    pairwise_iou = pairwise_iou.values

    paired_true = torch.nonzero(pairwise_iou > 0.0).squeeze(1)
    unpaired_true = torch.nonzero(pairwise_iou == 0.0).squeeze(1) + 1
    paired_pred = paired_pred[paired_true]

    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum().item()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum().item()

    pred_paired = torch.ones(len(pred_counts), dtype=torch.bool)
    pred_paired[paired_pred + 1] = False
    pred_paired[0] = False
    unpaired_pred = torch.nonzero(pred_paired).squeeze(1)

    overall_union += true_counts[unpaired_true].sum().item()
    overall_union += pred_counts[unpaired_pred].sum().item()

    aji_score = overall_inter / overall_union
    return aji_score


def _pq_impl(true: Tensor, pred: Tensor, match_iou=0.5):
    assert match_iou >= 0.0, "Cant' be negative"
    true, pred = true.to(torch.int).to(_device), pred.to(torch.int).to(_device)

    true_counts = torch.bincount(true.reshape(-1).cpu()).to(_device)
    pred_counts = torch.bincount(pred.reshape(-1).cpu()).to(_device)
    tp = torch.stack([true, pred], dim=2)
    tp = tp[(true != 0) & (pred != 0)]
    tp, counts = tp.unique(dim=0, return_counts=True)
    pairwise_iou = torch.zeros((len(true_counts) - 1, len(pred_counts) - 1), dtype=torch.float32, device=_device)
    true_ids, pred_ids = tp[:, 0], tp[:, 1]
    pairwise_iou[true_ids - 1, pred_ids - 1] = counts / (true_counts[true_ids] + pred_counts[pred_ids] - counts + 1e-6)
    paired_iou = pairwise_iou[pairwise_iou > match_iou]

    tp = len(paired_iou)
    fp = len(pred_counts) - tp - 1
    fn = len(true_counts) - tp - 1

    dq = tp / (tp + 0.5 * (fp + fn) + 1e-6)
    sq = paired_iou.sum().item() / (tp + 1e-6)

    return dq, sq, dq * sq


def _dice2_impl(true: Tensor, pred: Tensor):
    true, pred = true.to(torch.int).to(_device), pred.to(torch.int).to(_device)
    true_counts = torch.bincount(true.reshape(-1).cpu()).to(_device)
    pred_counts = torch.bincount(pred.reshape(-1).cpu()).to(_device)
    tp = torch.stack([true, pred], dim=2)
    tp = tp[(true != 0) & (pred != 0)]
    tp, counts = tp.unique(dim=0, return_counts=True)
    true_ids, pred_ids = tp[:, 0], tp[:, 1]
    overall_total = (true_counts[true_ids] + pred_counts[pred_ids]).sum().item()
    overall_inter = counts.sum().item()
    return 2 * overall_inter / (overall_total + 1e-6)


# 计算mPQ
def _mpq_impl(true_inst: Tensor, true_cls: Tensor, pred_inst: Tensor, pred_cls: Tensor):

    true_cls_id_list = torch.unique(true_cls)[1:].tolist()
    mpq = []
    for cls in true_cls_id_list:
        true_inst_c = true_inst.clone()
        true_inst_c[true_cls != cls] = 0
        true_inst_c = remap_label_fast(true_inst_c)

        pred_inst_c = pred_inst.clone()
        pred_inst_c[pred_cls != cls] = 0
        pred_inst_c = remap_label_fast(pred_inst_c)

        _, _, pq = _pq_impl(true_inst_c, pred_inst_c)
        mpq.append(pq)
    return sum(mpq) / len(mpq)


@torch.no_grad()
def remap_label(label: Tensor):
    """
    Remap label, make the values of label from 0(background) to N, N is the number of instances.
    Args:
        label: Tensor, shape is (H, W).
    Return:
        Tensor: remapped label.
    """

    ids = torch.unique(label, sorted=False)[1:].tolist()
    new_pred = torch.zeros_like(label)
    for i, id in enumerate(ids):
        new_pred[label == id] = i + 1
    return new_pred


# remap_label 原地版
@torch.no_grad()
def remap_label_(label: Tensor):
    """
    Remap label, make the values of label from 0(background) to N, N is the number of instances. This function will change the input label.
    Args:
        label: Tensor, shape is (H, W).
    Return:
        Tensor: remapped label, it is the same object with input label.
    """
    ids = torch.unique(label, sorted=False)[1:].tolist()
    for i, id in enumerate(ids):
        label[label == id] = i + 1
    return label


@torch.no_grad()
def remap_label_fast(label: Tensor, max_id: int = None):
    """
    If you ensure that the label‘s max value is not too big (like 1e6), you can use this function to remap label, it will be more faster than remap_label.
    Args:
        label: Tensor, shape is (H, W).
        max_id: int, the max value of label. If None, it will be label.max().
    Return:
        Tensor: remapped label.
    """

    if max_id is None:
        max_id = label.max().item()
    ids = label.unique()
    temp = torch.zeros(max_id + 1, dtype=label.dtype, device=label.device)
    temp[ids] = torch.arange(len(ids), dtype=label.dtype, device=label.device)
    return temp[label]
