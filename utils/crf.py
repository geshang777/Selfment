import numpy as np
import torch
import torch.nn.functional as F

MAX_ITER = 20
POS_W = 7
POS_XY_STD = 2
BI_W = 10
BI_XY_STD = 20
BI_RGB_STD = 5


def dense_crf(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    try:
        import pydensecrf.densecrf as dcrf
        import pydensecrf.utils as crf_utils
    except Exception as exc:
        raise RuntimeError(
            "pydensecrf is not installed. Install it to use CRF postprocess."
        ) from exc

    if mask.ndim != 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    mask_f = mask.astype(np.float32)
    if mask_f.max(initial=0.0) > 1.0:
        mask_f = mask_f / 255.0
    mask_f = np.clip(mask_f, 0.0, 1.0)

    h, w = mask_f.shape
    mask_f = mask_f.reshape(1, h, w)
    fg = mask_f.astype(np.float64)
    bg = 1 - fg
    output_logits = torch.from_numpy(np.concatenate((bg, fg), axis=0))

    h_img, w_img = image.shape[:2]
    output_logits = F.interpolate(
        output_logits.unsqueeze(0),
        size=(h_img, w_img),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()

    c, h_img, w_img = output_probs.shape
    u = crf_utils.unary_from_softmax(output_probs)
    u = np.ascontiguousarray(u)

    d = dcrf.DenseCRF2D(w_img, h_img, c)
    d.setUnaryEnergy(u)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=BI_XY_STD, srgb=BI_RGB_STD, rgbim=image, compat=BI_W)

    q = d.inference(MAX_ITER)
    q = np.array(q).reshape((c, h_img, w_img))
    out = np.argmax(q, axis=0).astype(np.float64)
    return out
