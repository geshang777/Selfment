import numpy as np
import torch


def iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    mask1, mask2 = (mask1 > 0.5).to(torch.bool), (mask2 > 0.5).to(torch.bool)
    intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
    union = torch.sum(mask1 + mask2, dim=[-1, -2]).squeeze()
    union_f = union.to(torch.float)
    inter_f = intersection.to(torch.float)
    iou_per = torch.where(union_f == 0, torch.ones_like(union_f), inter_f / union_f)
    return iou_per.mean().item()


def accuracy(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    mask1, mask2 = (mask1 > 0.5).to(torch.bool), (mask2 > 0.5).to(torch.bool)
    return torch.mean((mask1 == mask2).to(torch.float)).item()


class FMeasureTorch:
    def __init__(
        self,
        default_thres: float = 0.5,
        beta_square: float = 0.3,
        n_bins: int = 255,
        eps: float = 1e-7,
    ) -> None:
        self.beta_square = beta_square
        self.default_thres = default_thres
        self.eps = eps
        self.n_bins = n_bins

    def _compute_precision_recall(
        self, binary_pred_mask: torch.Tensor, gt_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tp = torch.logical_and(binary_pred_mask, gt_mask).sum(dim=(-1, -2))
        tp_fp = binary_pred_mask.sum(dim=(-1, -2))
        tp_fn = gt_mask.sum(dim=(-1, -2))
        prec = tp / (tp_fp + self.eps)
        recall = tp / (tp_fn + self.eps)
        return prec, recall

    def _compute_f_measure(
        self,
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        thresholds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if thresholds is None:
            binary_pred_mask = pred_mask > self.default_thres
        else:
            binary_pred_mask = pred_mask > thresholds
        prec, recall = self._compute_precision_recall(binary_pred_mask, gt_mask)
        f_measure = ((1 + (self.beta_square ** 2)) * prec * recall) / (
            (self.beta_square ** 2) * prec + recall + self.eps
        )
        return f_measure.cpu()

    def compute_f_max(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
        pred_masks = pred_mask.unsqueeze(dim=0).repeat(self.n_bins, 1, 1)
        gt_masks = gt_mask.unsqueeze(dim=0).repeat(self.n_bins, 1, 1)

        thresholds = (
            torch.arange(0, 1, 1 / self.n_bins)
            .view(self.n_bins, 1, 1)
            .to(pred_mask.device)
        )
        f_measures = self._compute_f_measure(pred_masks, gt_masks, thresholds)
        return float(torch.max(f_measures).item())


_FMEASURE_TORCH = FMeasureTorch()


@torch.no_grad()
def compute_simple_metrics(preds: list[np.ndarray], gts: list[np.ndarray]) -> dict:
    assert len(preds) == len(gts) and len(gts) > 0

    iou_sum = 0.0
    acc_sum = 0.0
    fmax_sum = 0.0

    for pred, gt in zip(preds, gts):
        pred_t = torch.from_numpy(pred).float()
        gt_t = torch.from_numpy(gt).float()

        iou_sum += iou(gt_t, pred_t)
        acc_sum += accuracy(gt_t, pred_t)

        gt_bin = gt_t > 0.5
        fmax_sum += _FMEASURE_TORCH.compute_f_max(pred_t, gt_bin)

    n = float(len(gts))
    return {
        "IoU": iou_sum / n,
        "accuracy": acc_sum / n,
        "F_max": fmax_sum / n,
    }


_EPS = np.spacing(1)
_TYPE = np.float64


def _prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gt = gt > 128
    pred = pred / 255.0
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt


def _get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1.0) -> float:
    return float(min(2 * matrix.mean(), max_value))


class FMeasure:
    def __init__(self, beta: float = 0.3) -> None:
        self.beta = beta
        self.precisions: list[np.ndarray] = []
        self.recalls: list[np.ndarray] = []
        self.adaptive_fms: list[float] = []
        self.changeable_fms: list[np.ndarray] = []

    def step(self, pred: np.ndarray, gt: np.ndarray) -> None:
        pred, gt = _prepare_data(pred, gt)
        self.adaptive_fms.append(self.cal_adaptive_fm(pred=pred, gt=gt))
        precisions, recalls, changeable_fms = self.cal_pr(pred=pred, gt=gt)
        self.precisions.append(precisions)
        self.recalls.append(recalls)
        self.changeable_fms.append(changeable_fms)

    def cal_adaptive_fm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        adaptive_threshold = _get_adaptive_threshold(pred, max_value=1.0)
        binary_predcition = pred >= adaptive_threshold
        area_intersection = binary_predcition[gt].sum()
        if area_intersection == 0:
            return 0.0
        pre = area_intersection / np.count_nonzero(binary_predcition)
        rec = area_intersection / np.count_nonzero(gt)
        return float((1 + self.beta) * pre * rec / (self.beta * pre + rec))

    def cal_pr(self, pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred_uint8 = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_hist, _ = np.histogram(pred_uint8[gt], bins=bins)
        bg_hist, _ = np.histogram(pred_uint8[~gt], bins=bins)
        fg_w_thrs = np.cumsum(np.flip(fg_hist), axis=0)
        bg_w_thrs = np.cumsum(np.flip(bg_hist), axis=0)
        tps = fg_w_thrs
        ps = fg_w_thrs + bg_w_thrs
        ps[ps == 0] = 1
        t = max(np.count_nonzero(gt), 1)
        precisions = tps / ps
        recalls = tps / t
        numerator = (1 + self.beta) * precisions * recalls
        denominator = np.where(numerator == 0, 1, self.beta * precisions + recalls)
        changeable_fms = numerator / denominator
        return precisions, recalls, changeable_fms

    def get_results(self) -> dict:
        adaptive_fm = np.mean(np.array(self.adaptive_fms, _TYPE))
        changeable_fm = np.mean(np.array(self.changeable_fms, dtype=_TYPE), axis=0)
        return dict(fm=dict(adp=float(adaptive_fm), curve=changeable_fm))


class MAEMeasure:
    def __init__(self) -> None:
        self.maes: list[float] = []

    def step(self, pred: np.ndarray, gt: np.ndarray) -> None:
        pred, gt = _prepare_data(pred, gt)
        self.maes.append(float(np.mean(np.abs(pred - gt))))

    def get_results(self) -> dict:
        mae = np.mean(np.array(self.maes, _TYPE))
        return dict(mae=float(mae))


class SMeasure:
    def __init__(self, alpha: float = 0.5) -> None:
        self.sms: list[float] = []
        self.alpha = alpha

    def step(self, pred: np.ndarray, gt: np.ndarray) -> None:
        pred, gt = _prepare_data(pred=pred, gt=gt)
        self.sms.append(self.cal_sm(pred, gt))

    def cal_sm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        y = np.mean(gt)
        if y == 0:
            return float(1 - np.mean(pred))
        if y == 1:
            return float(np.mean(pred))
        sm = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
        return float(max(0.0, sm))

    def object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        u = np.mean(gt)
        return float(u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, 1 - gt))

    def s_object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x = np.mean(pred[gt == 1])
        sigma_x = np.std(pred[gt == 1], ddof=1)
        return float(2 * x / (np.power(x, 2) + 1 + sigma_x + _EPS))

    def region(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x, y = self.centroid(gt)
        part = self.divide_with_xy(pred, gt, x, y)
        w1, w2, w3, w4 = part["weight"]
        pred1, pred2, pred3, pred4 = part["pred"]
        gt1, gt2, gt3, gt4 = part["gt"]
        score = (
            w1 * self.ssim(pred1, gt1)
            + w2 * self.ssim(pred2, gt2)
            + w3 * self.ssim(pred3, gt3)
            + w4 * self.ssim(pred4, gt4)
        )
        return float(score)

    def centroid(self, matrix: np.ndarray) -> tuple[int, int]:
        h, w = matrix.shape
        area_object = np.count_nonzero(matrix)
        if area_object == 0:
            x = np.round(w / 2)
            y = np.round(h / 2)
        else:
            y, x = np.argwhere(matrix).mean(axis=0).round()
        return int(x) + 1, int(y) + 1

    def divide_with_xy(self, pred: np.ndarray, gt: np.ndarray, x: int, y: int) -> dict:
        h, w = gt.shape
        area = h * w
        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = 1 - w1 - w2 - w3
        return dict(
            gt=(gt[0:y, 0:x], gt[0:y, x:w], gt[y:h, 0:x], gt[y:h, x:w]),
            pred=(pred[0:y, 0:x], pred[0:y, x:w], pred[y:h, 0:x], pred[y:h, x:w]),
            weight=(w1, w2, w3, w4),
        )

    def ssim(self, pred: np.ndarray, gt: np.ndarray) -> float:
        h, w = pred.shape
        n = h * w
        x = np.mean(pred)
        y = np.mean(gt)
        sigma_x = np.sum((pred - x) ** 2) / (n - 1)
        sigma_y = np.sum((gt - y) ** 2) / (n - 1)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (n - 1)
        alpha = 4 * x * y * sigma_xy
        beta = (x**2 + y**2) * (sigma_x + sigma_y)
        if alpha != 0:
            return float(alpha / (beta + _EPS))
        if alpha == 0 and beta == 0:
            return 1.0
        return 0.0

    def get_results(self) -> dict:
        sm = np.mean(np.array(self.sms, dtype=_TYPE))
        return dict(sm=float(sm))


class EMeasure:
    def __init__(self) -> None:
        self.adaptive_ems: list[float] = []
        self.changeable_ems: list[np.ndarray] = []

    def step(self, pred: np.ndarray, gt: np.ndarray) -> None:
        pred, gt = _prepare_data(pred=pred, gt=gt)
        self.gt_fg_numel = np.count_nonzero(gt)
        self.gt_size = gt.shape[0] * gt.shape[1]
        self.changeable_ems.append(self.cal_em_with_cumsumhistogram(pred, gt))
        self.adaptive_ems.append(self.cal_adaptive_em(pred, gt))

    def cal_adaptive_em(self, pred: np.ndarray, gt: np.ndarray) -> float:
        adaptive_threshold = _get_adaptive_threshold(pred, max_value=1.0)
        return float(self.cal_em_with_threshold(pred, gt, threshold=adaptive_threshold))

    def cal_em_with_threshold(self, pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
        binarized_pred = pred >= threshold
        fg_fg_numel = np.count_nonzero(binarized_pred & gt)
        fg_bg_numel = np.count_nonzero(binarized_pred & ~gt)
        fg___numel = fg_fg_numel + fg_bg_numel
        bg___numel = self.gt_size - fg___numel
        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg___numel
        else:
            parts_numel, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel,
                fg_bg_numel=fg_bg_numel,
                pred_fg_numel=fg___numel,
                pred_bg_numel=bg___numel,
            )
            results_parts = []
            for part_numel, combination in zip(parts_numel, combinations):
                align_matrix_value = 2 * (combination[0] * combination[1]) / (
                    combination[0] ** 2 + combination[1] ** 2 + _EPS
                )
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts.append(enhanced_matrix_value * part_numel)
            enhanced_matrix_sum = sum(results_parts)
        return float(enhanced_matrix_sum / (self.gt_size - 1 + _EPS))

    def cal_em_with_cumsumhistogram(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        pred_uint8 = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_fg_hist, _ = np.histogram(pred_uint8[gt], bins=bins)
        fg_bg_hist, _ = np.histogram(pred_uint8[~gt], bins=bins)
        fg_fg_numel_w_thrs = np.cumsum(np.flip(fg_fg_hist), axis=0)
        fg_bg_numel_w_thrs = np.cumsum(np.flip(fg_bg_hist), axis=0)
        fg___numel_w_thrs = fg_fg_numel_w_thrs + fg_bg_numel_w_thrs
        bg___numel_w_thrs = self.gt_size - fg___numel_w_thrs
        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel_w_thrs
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg___numel_w_thrs
        else:
            parts_numel_w_thrs, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel_w_thrs,
                fg_bg_numel=fg_bg_numel_w_thrs,
                pred_fg_numel=fg___numel_w_thrs,
                pred_bg_numel=bg___numel_w_thrs,
            )
            results_parts = np.empty(shape=(4, 256), dtype=np.float64)
            for i, (part_numel, combination) in enumerate(
                zip(parts_numel_w_thrs, combinations)
            ):
                align_matrix_value = 2 * (combination[0] * combination[1]) / (
                    combination[0] ** 2 + combination[1] ** 2 + _EPS
                )
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts[i] = enhanced_matrix_value * part_numel
            enhanced_matrix_sum = results_parts.sum(axis=0)
        return enhanced_matrix_sum / (self.gt_size - 1 + _EPS)

    def generate_parts_numel_combinations(
        self,
        fg_fg_numel,
        fg_bg_numel,
        pred_fg_numel,
        pred_bg_numel,
    ):
        bg_fg_numel = self.gt_fg_numel - fg_fg_numel
        bg_bg_numel = pred_bg_numel - bg_fg_numel
        parts_numel = [fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel]
        mean_pred_value = pred_fg_numel / self.gt_size
        mean_gt_value = self.gt_fg_numel / self.gt_size
        demeaned_pred_fg_value = 1 - mean_pred_value
        demeaned_pred_bg_value = 0 - mean_pred_value
        demeaned_gt_fg_value = 1 - mean_gt_value
        demeaned_gt_bg_value = 0 - mean_gt_value
        combinations = [
            (demeaned_pred_fg_value, demeaned_gt_fg_value),
            (demeaned_pred_fg_value, demeaned_gt_bg_value),
            (demeaned_pred_bg_value, demeaned_gt_fg_value),
            (demeaned_pred_bg_value, demeaned_gt_bg_value),
        ]
        return parts_numel, combinations

    def get_results(self) -> dict:
        adaptive_em = np.mean(np.array(self.adaptive_ems, dtype=_TYPE))
        changeable_em = np.mean(np.array(self.changeable_ems, dtype=_TYPE), axis=0)
        return dict(em=dict(adp=float(adaptive_em), curve=changeable_em))


class WeightedFMeasure:
    def __init__(self, beta: float = 1.0) -> None:
        from scipy.ndimage import convolve

        self.beta = beta
        self.weighted_fms: list[float] = []
        self._convolve = convolve

    def step(self, pred: np.ndarray, gt: np.ndarray) -> None:
        pred, gt = _prepare_data(pred=pred, gt=gt)
        if np.all(~gt):
            self.weighted_fms.append(0.0)
            return
        self.weighted_fms.append(self.cal_wfm(pred, gt))

    def cal_wfm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        from scipy.ndimage import distance_transform_edt as bwdist

        dst, idxt = bwdist(gt == 0, return_indices=True)
        e = np.abs(pred - gt)
        et = np.copy(e)
        et[gt == 0] = et[idxt[0][gt == 0], idxt[1][gt == 0]]
        k = self.matlab_style_gauss2d((7, 7), sigma=5)
        ea = self._convolve(et, weights=k, mode="constant", cval=0)
        min_e_ea = np.where(gt & (ea < e), ea, e)
        b = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * dst), np.ones_like(gt))
        ew = min_e_ea * b
        tpw = np.sum(gt) - np.sum(ew[gt == 1])
        fpw = np.sum(ew[gt == 0])
        r = 1 - np.mean(ew[gt == 1])
        p = tpw / (tpw + fpw + _EPS)
        return float((1 + self.beta) * r * p / (r + self.beta * p + _EPS))

    def matlab_style_gauss2d(self, shape: tuple = (7, 7), sigma: int = 5) -> np.ndarray:
        m, n = [(ss - 1) / 2 for ss in shape]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def get_results(self) -> dict:
        weighted_fm = np.mean(np.array(self.weighted_fms, dtype=_TYPE))
        return dict(wfm=float(weighted_fm))


def compute_biref_metrics(preds: list[np.ndarray], gts: list[np.ndarray]) -> dict:
    assert len(preds) == len(gts) and len(gts) > 0

    measures = {
        "E": EMeasure(),
        "S": SMeasure(),
        "F": FMeasure(),
        "MAE": MAEMeasure(),
        "WF": WeightedFMeasure(),
    }

    for pred, gt in zip(preds, gts):
        pred_arr = (pred * 255.0).astype(np.float32)
        gt_arr = (gt * 255.0).astype(np.float32)
        measures["E"].step(pred=pred_arr, gt=gt_arr)
        measures["S"].step(pred=pred_arr, gt=gt_arr)
        measures["F"].step(pred=pred_arr, gt=gt_arr)
        measures["MAE"].step(pred=pred_arr, gt=gt_arr)
        measures["WF"].step(pred=pred_arr, gt=gt_arr)

    em = measures["E"].get_results()["em"]
    sm = measures["S"].get_results()["sm"]
    fm = measures["F"].get_results()["fm"]
    mae = measures["MAE"].get_results()["mae"]
    wfm = measures["WF"].get_results()["wfm"]

    em_curve = em["curve"]
    fm_curve = fm["curve"]

    return {
        "wFmeasure": float(wfm),
        "MAE": float(mae),
        "Smeasure": float(sm),
        "meanEm": float(np.mean(em_curve)),
        "maxEm": float(np.max(em_curve)),
        "adpEm": float(em["adp"]),
        "meanFm": float(np.mean(fm_curve)),
        "maxFm": float(np.max(fm_curve)),
        "adpFm": float(fm["adp"]),
    }


def compute_all_metrics(preds: list[np.ndarray], gts: list[np.ndarray]) -> dict:
    out = {}
    out.update(compute_simple_metrics(preds, gts))
    out.update(compute_biref_metrics(preds, gts))
    return out
