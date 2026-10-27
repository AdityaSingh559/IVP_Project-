"""Core image-processing primitives, written from scratch on NumPy.

Everything here follows the textbook definition (Gonzalez & Woods). OpenCV is used elsewhere
only for the extension techniques (Hough, bilateral, CLAHE, inpainting) and for Lab conversion.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage

# ----------------------------------------------------------------------------- basics


def to_u8(x):
    """Float image in 0..255 -> uint8 with rounding and clipping."""
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def rgb_to_gray(rgb):
    """ITU-R BT.601 luminance of an RGB image (float result, 0..255)."""
    rgb = rgb.astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


# ----------------------------------------------------------------------------- colour models


def rgb_to_hsv(rgb):
    """rgb: float 0..1 (H,W,3) -> h in 0..1, s, v in 0..1."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    d = mx - mn
    s = np.where(mx > 1e-8, d / np.maximum(mx, 1e-8), 0.0)
    dd = np.maximum(d, 1e-8)
    h = np.zeros_like(mx)
    h = np.where(mx == r, ((g - b) / dd) % 6.0, h)
    h = np.where(mx == g, (b - r) / dd + 2.0, h)
    h = np.where(mx == b, (r - g) / dd + 4.0, h)
    h = np.where(d < 1e-8, 0.0, h) / 6.0
    return np.stack([h % 1.0, s, mx], axis=-1).astype(np.float32)


def hsv_to_rgb(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def rgb_to_yuv(rgb):
    """rgb float (any scale) -> Y, U, V (analogue BT.601)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.stack([y, 0.492 * (b - y), 0.877 * (r - y)], axis=-1).astype(np.float32)


def yuv_to_rgb(yuv):
    y, u, v = yuv[..., 0], yuv[..., 1], yuv[..., 2]
    r = y + v / 0.877
    b = y + u / 0.492
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def lab_distance(a_rgb_u8, b_rgb_u8):
    """Mean Euclidean distance between two images in CIE Lab (a perceptual colour distance)."""
    import cv2

    la = cv2.cvtColor(a_rgb_u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    lb = cv2.cvtColor(b_rgb_u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    return float(np.sqrt(((la - lb) ** 2).sum(axis=-1)).mean())


# ----------------------------------------------------------------------------- filtering


def gaussian_kernel1d(sigma):
    radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def _sep_filter(img2d, k1d):
    r = len(k1d) // 2
    p = np.pad(img2d, ((r, r), (0, 0)), mode="reflect")
    tmp = (sliding_window_view(p, len(k1d), axis=0) * k1d).sum(axis=-1)
    p = np.pad(tmp, ((0, 0), (r, r)), mode="reflect")
    return (sliding_window_view(p, len(k1d), axis=1) * k1d).sum(axis=-1)


def gaussian_blur(img, sigma):
    """Gaussian smoothing (separable). Works on (H,W) or (H,W,C) float arrays."""
    img = img.astype(np.float32)
    k = gaussian_kernel1d(sigma)
    if img.ndim == 2:
        return _sep_filter(img, k)
    return np.stack([_sep_filter(img[..., c], k) for c in range(img.shape[-1])], axis=-1)


def convolve2d(img, kernel):
    """True 2-D convolution with reflected borders."""
    kh, kw = kernel.shape
    p = np.pad(img.astype(np.float32), ((kh // 2,) * 2, (kw // 2,) * 2), mode="reflect")
    win = sliding_window_view(p, (kh, kw))
    return np.einsum("ijkl,kl->ij", win, kernel[::-1, ::-1].astype(np.float32))


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = SOBEL_X.T
LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def sobel(gray):
    gx = convolve2d(gray, SOBEL_X)
    gy = convolve2d(gray, SOBEL_Y)
    return gx, gy, np.hypot(gx, gy)


def laplacian(gray):
    return convolve2d(gray, LAPLACIAN)


def median3x3(ch):
    """3x3 median of a single channel."""
    p = np.pad(ch, 1, mode="reflect")
    h, w = ch.shape
    stack = np.stack([p[i : i + h, j : j + w] for i in range(3) for j in range(3)], axis=0)
    return np.partition(stack, 4, axis=0)[4]


# ----------------------------------------------------------------------------- histograms / point operations


def histogram(u8):
    return np.bincount(u8.ravel(), minlength=256)


def equalize(u8):
    """Global histogram equalization of a uint8 channel."""
    cdf = histogram(u8).cumsum()
    nz = cdf[cdf > 0]
    cdf_min = nz[0]
    denom = max(int(cdf[-1] - cdf_min), 1)
    lut = np.clip(np.rint((cdf - cdf_min) / denom * 255), 0, 255).astype(np.uint8)
    return lut[u8]


def match_histogram(src_u8, ref_u8):
    """Histogram specification: remap src so its histogram follows ref's."""
    s_cdf = histogram(src_u8).cumsum() / src_u8.size
    r_cdf = histogram(ref_u8).cumsum() / ref_u8.size
    lut = np.searchsorted(r_cdf, s_cdf).clip(0, 255).astype(np.uint8)
    return lut[src_u8]


def contrast_stretch(x, lo, hi):
    """Piecewise-linear contrast stretching of x (any scale) from [lo,hi] to [0,1]."""
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def gamma_transform(x01, gamma):
    return np.power(np.clip(x01, 0, 1), gamma)


def log_transform(x01, c=5.0):
    return np.log1p(c * np.clip(x01, 0, 1)) / np.log1p(c)


def otsu_threshold(u8):
    """Otsu's method: threshold that maximises between-class variance."""
    hist = histogram(u8).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0
    p = hist / total
    omega = p.cumsum()
    mu = (p * np.arange(256)).cumsum()
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    sigma_b = np.where(denom > 1e-12, (mu_t * omega - mu) ** 2 / np.maximum(denom, 1e-12), 0)
    return int(np.argmax(sigma_b))


# ----------------------------------------------------------------------------- morphology (grey-scale, square SE)


def _slide(img, k, func):
    r = k // 2
    p = np.pad(img, ((r, r), (0, 0)), mode="edge")
    tmp = func(sliding_window_view(p, k, axis=0), axis=-1)
    p = np.pad(tmp, ((0, 0), (r, r)), mode="edge")
    return func(sliding_window_view(p, k, axis=1), axis=-1)


def erode(img, k):
    return _slide(img, k, np.min)


def dilate(img, k):
    return _slide(img, k, np.max)


def opening(img, k):
    return dilate(erode(img, k), k)


def closing(img, k):
    return erode(dilate(img, k), k)


def top_hat(img, k):
    """Bright details thinner than the structuring element."""
    return img - opening(img, k)


def black_hat(img, k):
    """Dark details thinner than the structuring element."""
    return closing(img, k) - img


def box_filter(img, r):
    """Mean over a (2r+1) x (2r+1) window, reflected borders."""
    return ndimage.uniform_filter(img.astype(np.float32), size=2 * r + 1, mode="reflect")


def guided_filter(guide, src, r, eps):
    """Guided filter (He et al.): edge-preserving smoothing of `src` that follows the edges of
    `guide`. Inside every window the output is a linear function of the guide, q = a*I + b,
    with a and b chosen to minimise the squared error to `src` plus eps*a^2."""
    I, p = guide.astype(np.float32), src.astype(np.float32)
    mean_i, mean_p = box_filter(I, r), box_filter(p, r)
    var_i = box_filter(I * I, r) - mean_i * mean_i
    cov_ip = box_filter(I * p, r) - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box_filter(a, r) * I + box_filter(b, r)


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, dtype=np.float32) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def line_close(mask, r, dy, dx):
    """Binary closing with a straight line structuring element of length 2r+1 pointing along
    (dy, dx). Bridges gaps in thin, broken lines without thickening them."""
    h, w = mask.shape
    p = np.pad(mask, r, mode="constant", constant_values=False)
    dil = np.zeros_like(mask)
    for t in range(-r, r + 1):
        dil |= p[r + t * dy : r + t * dy + h, r + t * dx : r + t * dx + w]
    p = np.pad(dil, r, mode="constant", constant_values=True)
    out = np.ones_like(mask)
    for t in range(-r, r + 1):
        out &= p[r + t * dy : r + t * dy + h, r + t * dx : r + t * dx + w]
    return out


def line_ridge(gray, length=9, offset=3):
    """Response to thin bright lines in any of eight directions.

    For each direction the image is averaged along a short line through the pixel, and the
    result is compared with parallel lines `offset` pixels away on both sides. A crease is
    brighter than BOTH neighbours, so the response is the smaller of the two differences. An
    object edge is brighter on one side only, so it scores near zero."""
    g = gaussian_blur(gray.astype(np.float32), 0.8)
    h, w = g.shape
    r = length // 2
    pad = r + offset + 2
    p = np.pad(g, pad, mode="reflect")
    best = np.zeros_like(g)
    for ang in np.arange(0, 180, 22.5):
        c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        offs = [(int(round(t * s)), int(round(t * c))) for t in range(-r, r + 1)]

        def line_mean(oy, ox):
            acc = np.zeros_like(g)
            for dy, dx in offs:
                acc += p[pad + oy + dy : pad + oy + dy + h, pad + ox + dx : pad + ox + dx + w]
            return acc / len(offs)

        oy, ox = int(round(offset * c)), int(round(-offset * s))
        centre = line_mean(0, 0)
        best = np.maximum(best, np.minimum(centre - line_mean(oy, ox), centre - line_mean(-oy, -ox)))
    return np.maximum(best, 0)


def long_line_pixels(mask, r=10):
    """Pixels of a binary mask that lie on a straight run at least 2r+1 pixels long, in any of
    eight directions: the union of openings with line-shaped structuring elements. Cracks and
    scratches contain such runs; small blobs and highlights do not."""
    h, w = mask.shape
    out = np.zeros_like(mask)
    pad = r + 1
    for ang in np.arange(0, 180, 22.5):
        c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        offs = [(int(round(t * s)), int(round(t * c))) for t in range(-r, r + 1)]
        p = np.pad(mask, pad)
        er = np.ones_like(mask)
        for dy, dx in offs:
            er &= p[pad + dy : pad + dy + h, pad + dx : pad + dx + w]
        pe = np.pad(er, pad)
        di = np.zeros_like(mask)
        for dy, dx in offs:
            di |= pe[pad - dy : pad - dy + h, pad - dx : pad - dx + w]
        out |= di
    return out & mask


def label_components(mask):
    """8-connected component labelling."""
    return ndimage.label(mask, structure=np.ones((3, 3), dtype=int))


# ----------------------------------------------------------------------------- Canny


def canny(gray, sigma=1.4, low=0.08, high=0.2):
    """Canny edge detector: Gaussian smoothing, Sobel gradient, non-maximum suppression,
    double threshold with hysteresis (thresholds are fractions of the maximum gradient)."""
    g = gaussian_blur(gray.astype(np.float32) / 255.0, sigma)
    gx, gy, mag = sobel(g)
    ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
    q = ((ang + 22.5) // 45).astype(np.int32) % 4  # 0:0deg 1:45deg 2:90deg 3:135deg
    p = np.pad(mag, 1)
    c = p[1:-1, 1:-1]
    pairs = {
        0: (p[1:-1, 2:], p[1:-1, :-2]),
        1: (p[2:, 2:], p[:-2, :-2]),
        2: (p[2:, 1:-1], p[:-2, 1:-1]),
        3: (p[2:, :-2], p[:-2, 2:]),
    }
    keep = np.zeros(mag.shape, dtype=bool)
    for d, (n1, n2) in pairs.items():
        keep |= (q == d) & (c >= n1) & (c >= n2)
    thin = np.where(keep, mag, 0)
    mx = thin.max()
    if mx <= 0:
        return np.zeros(mag.shape, dtype=bool)
    strong = thin >= high * mx
    weak = thin >= low * mx
    labels, n = label_components(weak)
    ok = np.zeros(n + 1, dtype=bool)
    ok[labels[strong]] = True
    ok[0] = False
    return ok[labels]


# ----------------------------------------------------------------------------- transforms


def dct_matrix(n=8):
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    c = np.cos((2 * i + 1) * k * np.pi / (2 * n))
    c[0] *= 1 / np.sqrt(2)
    return (c * np.sqrt(2 / n)).astype(np.float32)


_C8 = dct_matrix(8)


def dct_shrink(ch, sigma, thr_mult=2.7):
    """DCT-domain denoising: 8x8 block DCT, hard-threshold small AC coefficients, inverse DCT.
    Four grid shifts are averaged so no block seams remain."""
    h, w = ch.shape
    acc = np.zeros((h, w), dtype=np.float32)
    thr = thr_mult * sigma
    shifts = [(0, 0), (4, 0), (0, 4), (4, 4)]
    for dy, dx in shifts:
        ph, pw = (-(h + dy)) % 8, (-(w + dx)) % 8
        p = np.pad(ch, ((dy, ph), (dx, pw)), mode="reflect").astype(np.float32)
        H, W = p.shape
        blocks = p.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3)
        coef = np.matmul(np.matmul(_C8, blocks), _C8.T)
        dc = coef[..., 0, 0].copy()
        coef[np.abs(coef) < thr] = 0
        coef[..., 0, 0] = dc
        rec = np.matmul(np.matmul(_C8.T, coef), _C8)
        rec = rec.transpose(0, 2, 1, 3).reshape(H, W)
        acc += rec[dy : dy + h, dx : dx + w]
    return acc / len(shifts)


def fft_spectrum_image(gray):
    """Centred log-magnitude Fourier spectrum scaled to uint8."""
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32) - gray.mean()))
    mag = np.log1p(np.abs(f))
    mag = (mag - mag.min()) / max(mag.max() - mag.min(), 1e-6)
    return to_u8(mag * 255)


def homomorphic(l01, gamma_l, gamma_h, d0, c=1.0, pad=48):
    """Homomorphic filter on a 0..1 luminance image. Illumination (low frequency) is
    attenuated by gamma_l, reflectance detail (high frequency) is boosted by gamma_h."""
    z = np.log(np.clip(l01, 0, 1) + 0.01)
    zp = np.pad(z, pad, mode="reflect")
    hh, ww = zp.shape
    u = np.fft.fftfreq(hh)[:, None]
    v = np.fft.fftfreq(ww)[None, :]
    d2 = u**2 + v**2
    H = (gamma_h - gamma_l) * (1 - np.exp(-c * d2 / (d0**2))) + gamma_l
    out = np.real(np.fft.ifft2(np.fft.fft2(zp) * H))[pad:-pad, pad:-pad]
    return np.exp(out) - 0.01, np.fft.fftshift(H)


def wiener_deblur(ch, sigma_psf, k):
    """Wiener deconvolution of a single channel assuming a Gaussian blur of std sigma_psf."""
    pad = int(np.ceil(4 * sigma_psf)) + 8
    p = np.pad(ch.astype(np.float32), pad, mode="reflect")
    hh, ww = p.shape
    yy = np.fft.fftfreq(hh)[:, None]
    xx = np.fft.fftfreq(ww)[None, :]
    H = np.exp(-2 * (np.pi**2) * (sigma_psf**2) * (xx**2 + yy**2))
    G = np.fft.fft2(p)
    F = np.conj(H) / (np.abs(H) ** 2 + k) * G
    return np.real(np.fft.ifft2(F))[pad:-pad, pad:-pad]
