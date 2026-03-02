from __future__ import annotations

import numpy as np
import PIL.Image as Image
from scipy import ndimage
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg

RGB_TO_YUV = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ]
)
YUV_TO_RGB = np.array(
    [[1.0, 0.0, 1.402], [1.0, -0.34414, -0.71414], [1.0, 1.772, 0.0]])
YUV_OFFSET = np.array([0, 128.0, 128.0]).reshape(1, 1, -1)
MAX_VAL = 255.0


def rgb2yuv(im: np.ndarray) -> np.ndarray:
    return np.tensordot(im, RGB_TO_YUV, ([2], [1])) + YUV_OFFSET


def yuv2rgb(im: np.ndarray) -> np.ndarray:
    return np.tensordot(im.astype(float) - YUV_OFFSET, YUV_TO_RGB, ([2], [1]))


def get_valid_idx(valid: np.ndarray, candidates: np.ndarray):
    locs = np.searchsorted(valid, candidates)
    locs = np.clip(locs, 0, len(valid) - 1)
    valid_idx = np.flatnonzero(valid[locs] == candidates)
    locs = locs[valid_idx]
    return valid_idx, locs


class BilateralGrid:
    def __init__(self, im: np.ndarray, sigma_spatial=32, sigma_luma=8, sigma_chroma=8):
        im_yuv = rgb2yuv(im)
        iy, ix = np.mgrid[: im.shape[0], : im.shape[1]]
        x_coords = (ix / sigma_spatial).astype(int)
        y_coords = (iy / sigma_spatial).astype(int)
        luma_coords = (im_yuv[..., 0] / sigma_luma).astype(int)
        chroma_coords = (im_yuv[..., 1:] / sigma_chroma).astype(int)
        coords = np.dstack((x_coords, y_coords, luma_coords, chroma_coords))
        coords_flat = coords.reshape(-1, coords.shape[-1])
        self.npixels, self.dim = coords_flat.shape
        self.hash_vec = MAX_VAL ** np.arange(self.dim)
        self._compute_factorization(coords_flat)

    def _compute_factorization(self, coords_flat: np.ndarray):
        hashed_coords = self._hash_coords(coords_flat)
        unique_hashes, unique_idx, idx = np.unique(
            hashed_coords, return_index=True, return_inverse=True
        )
        unique_coords = coords_flat[unique_idx]
        self.nvertices = len(unique_coords)
        self.S = csr_matrix((np.ones(self.npixels), (idx, np.arange(self.npixels))))
        self.blurs = []
        for d in range(self.dim):
            blur = 0.0
            for offset in (-1, 1):
                offset_vec = np.zeros((1, self.dim))
                offset_vec[:, d] = offset
                neighbor_hash = self._hash_coords(unique_coords + offset_vec)
                valid_coord, idx2 = get_valid_idx(unique_hashes, neighbor_hash)
                blur = blur + csr_matrix(
                    (np.ones((len(valid_coord),)), (valid_coord, idx2)),
                    shape=(self.nvertices, self.nvertices),
                )
            self.blurs.append(blur)

    def _hash_coords(self, coord: np.ndarray) -> np.ndarray:
        return np.dot(coord.reshape(-1, self.dim), self.hash_vec)

    def splat(self, x: np.ndarray) -> np.ndarray:
        return self.S.dot(x)

    def slice(self, y: np.ndarray) -> np.ndarray:
        return self.S.T.dot(y)

    def blur(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[0] == self.nvertices
        out = 2 * self.dim * x
        for blur in self.blurs:
            out = out + blur.dot(x)
        return out


def bistochastize(grid: BilateralGrid, maxiter=10):
    m = grid.splat(np.ones(grid.npixels))
    n = np.ones(grid.nvertices)
    for _ in range(maxiter):
        n = np.sqrt(n * m / grid.blur(n))
    m = n * grid.blur(n)
    dm = diags(m, 0)
    dn = diags(n, 0)
    return dn, dm


class BilateralSolver:
    def __init__(self, grid: BilateralGrid, params: dict):
        self.grid = grid
        self.params = params
        self.Dn, self.Dm = bistochastize(grid)

    def solve(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        if w.ndim == 2:
            assert w.shape[1] == 1
        elif w.ndim == 1:
            w = w.reshape(w.shape[0], 1)
        a_smooth = self.Dm - self.Dn.dot(self.grid.blur(self.Dn))
        w_splat = self.grid.splat(w)
        a_data = diags(w_splat[:, 0], 0)
        a = self.params["lam"] * a_smooth + a_data
        xw = x * w
        b = self.grid.splat(xw)
        a_diag = np.maximum(a.diagonal(), self.params["A_diag_min"])
        m = diags(1 / a_diag, 0)
        y0 = self.grid.splat(xw) / w_splat
        yhat = np.empty_like(y0)
        for d in range(x.shape[-1]):
            yhat[..., d], _ = cg(
                a,
                b[..., d],
                x0=y0[..., d],
                M=m,
                maxiter=self.params["cg_maxiter"],
                atol=self.params["cg_tol"],
            )
        xhat = self.grid.slice(yhat)
        return xhat


def bilateral_solver_output(
    img_path: str,
    target: np.ndarray,
    sigma_spatial=24,
    sigma_luma=4,
    sigma_chroma=4,
):
    reference = np.array(Image.open(img_path).convert("RGB"))
    h, w = target.shape
    confidence = np.ones((h, w)) * 0.999

    grid_params = {
        "sigma_luma": sigma_luma,
        "sigma_chroma": sigma_chroma,
        "sigma_spatial": sigma_spatial,
    }

    bs_params = {
        "lam": 512,
        "A_diag_min": 1e-5,
        "cg_tol": 1e-5,
        "cg_maxiter": 25,
    }

    grid = BilateralGrid(reference, **grid_params)
    t = target.reshape(-1, 1).astype(np.double)
    c = confidence.reshape(-1, 1).astype(np.double)

    output_solver = BilateralSolver(grid, bs_params).solve(t, c).reshape((h, w))

    binary_solver = ndimage.binary_fill_holes(output_solver > 0.5)
    labeled, nr_objects = ndimage.label(binary_solver)

    nb_pixel = [np.sum(labeled == i) for i in range(nr_objects + 1)]
    pixel_order = np.argsort(nb_pixel)
    try:
        binary_solver = labeled == pixel_order[-2]
    except Exception:
        binary_solver = np.ones((h, w), dtype=bool)

    return output_solver, binary_solver

