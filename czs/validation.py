"""Validate predicted zones against agronomist ground-truth markings."""
import numpy as np
from scipy.ndimage import distance_transform_edt


def confusion(pred, truth, n=3):
    cm = np.zeros((n, n), int)
    np.add.at(cm, (truth.ravel(), pred.ravel()), 1)
    return cm  # rows = agronomist, cols = predicted


def metrics(pred, truth, n=3):
    cm = confusion(pred, truth, n)
    total = cm.sum()
    oa = np.trace(cm) / total
    pe = (cm.sum(0) * cm.sum(1)).sum() / total ** 2
    kappa = (oa - pe) / (1 - pe + 1e-12)
    per = []
    for c in range(n):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + 1e-12); rec = tp / (tp + fn + 1e-12)
        per.append(dict(cls=c, precision=prec, recall=rec,
                        f1=2 * prec * rec / (prec + rec + 1e-12),
                        iou=tp / (tp + fp + fn + 1e-12)))
    return dict(cm=cm, overall_accuracy=oa, kappa=kappa,
                miou=float(np.mean([p["iou"] for p in per])), per_class=per)


def boundary_map(lbl):
    b = np.zeros(lbl.shape, bool)
    b[:, :-1] |= lbl[:, :-1] != lbl[:, 1:]
    b[:-1, :] |= lbl[:-1, :] != lbl[1:, :]
    return b


def boundary_f1(pred, truth, tol=3):
    """Boundary F-score: boundary pixels matched within `tol` pixels."""
    bp, bt = boundary_map(pred), boundary_map(truth)
    if bp.sum() == 0 or bt.sum() == 0:
        return dict(precision=0.0, recall=0.0, f1=0.0)
    dt_t = distance_transform_edt(~bt); dt_p = distance_transform_edt(~bp)
    prec = float((dt_t[bp] <= tol).mean()); rec = float((dt_p[bt] <= tol).mean())
    return dict(precision=prec, recall=rec, f1=2 * prec * rec / (prec + rec + 1e-12))
