# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Image Processing Toolbox for Forge — Octave-compatible functions.

Implements 58 Octave image functions: colormaps, image I/O, color conversion,
colormap operations, and basic image processing.

Backend: NumPy + Pillow (PIL) + matplotlib.cm for colormaps.

SRS trace: SRS-FUNC-IMAGE
"""

from __future__ import annotations

import warnings
import numpy as np

# ── Optional heavy imports (lazy) ────────────────────────────────
try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None

try:
    import matplotlib.cm as _mpl_cm
    import matplotlib.colors as _mpl_colors
except ImportError:
    _mpl_cm = None
    _mpl_colors = None

# ── ForgeArray interop ───────────────────────────────────────────
try:
    from forge.engine.types import ForgeArray, _unwrap
except ImportError:
    ForgeArray = np.ndarray

    def _unwrap(x):
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x, dtype=np.float64)

try:
    from forge.engine.containers import ForgeChar
except ImportError:
    ForgeChar = str


def _fa(x):
    """Wrap result as ForgeArray."""
    arr = np.asarray(x)
    if ForgeArray is np.ndarray:
        return arr
    return ForgeArray(arr)


def _ensure_float(x):
    return np.asarray(_unwrap(x), dtype=np.float64)


def _scalar(x):
    if isinstance(x, ForgeArray):
        d = x.data if hasattr(x, 'data') else np.asarray(x)
        return d.flat[0].item() if d.size == 1 else d
    if isinstance(x, np.ndarray) and x.size == 1:
        return x.flat[0].item()
    return x


# ═══════════════════════════════════════════════════════════════════
# 1. COLORMAPS  (return Nx3 float64 arrays, values in [0,1])
# ═══════════════════════════════════════════════════════════════════

def _cmap_from_mpl(name, N=256):
    """Generate an Nx3 colormap array from matplotlib."""
    N = int(N)
    if _mpl_cm is None:
        raise RuntimeError("matplotlib is required for colormap functions")
    cm = _mpl_cm.get_cmap(name, N)
    return _fa(cm(np.linspace(0, 1, N))[:, :3])


def autumn(N=256):
    """Autumn colormap. MAP = autumn(N) returns an Nx3 array."""
    return _cmap_from_mpl('autumn', N)


def bone(N=256):
    """Bone colormap. MAP = bone(N) returns an Nx3 array."""
    return _cmap_from_mpl('bone', N)


def cool(N=256):
    """Cool colormap. MAP = cool(N) returns an Nx3 array."""
    return _cmap_from_mpl('cool', N)


def copper(N=256):
    """Copper colormap. MAP = copper(N) returns an Nx3 array."""
    return _cmap_from_mpl('copper', N)


def cubehelix(N=256):
    """Cubehelix colormap. MAP = cubehelix(N) returns an Nx3 array."""
    return _cmap_from_mpl('cubehelix', N)


def flag(N=256):
    """Flag colormap. MAP = flag(N) returns an Nx3 array."""
    return _cmap_from_mpl('flag', N)


def gray(N=256):
    """Gray (linear grayscale) colormap. MAP = gray(N) returns an Nx3 array."""
    return _cmap_from_mpl('gray', N)


def hot(N=256):
    """Hot colormap. MAP = hot(N) returns an Nx3 array."""
    return _cmap_from_mpl('hot', N)


def hsv(N=256):
    """HSV colormap. MAP = hsv(N) returns an Nx3 array."""
    return _cmap_from_mpl('hsv', N)


def jet(N=256):
    """Jet colormap. MAP = jet(N) returns an Nx3 array."""
    return _cmap_from_mpl('jet', N)


def lines(N=256):
    """Lines colormap (default axes color order, cycled).

    MAP = lines(N) returns an Nx3 array.
    """
    N = int(N)
    # Octave 'lines' is the default color-order cycled.  Use tab10.
    if _mpl_cm is None:
        raise RuntimeError("matplotlib is required for colormap functions")
    base = _mpl_cm.get_cmap('tab10', 10)
    colors = base(np.linspace(0, 1, 10))[:, :3]
    idx = np.arange(N) % 10
    return _fa(colors[idx])


def ocean(N=256):
    """Ocean colormap. MAP = ocean(N) returns an Nx3 array."""
    return _cmap_from_mpl('ocean', N)


def pink(N=256):
    """Pink colormap. MAP = pink(N) returns an Nx3 array."""
    return _cmap_from_mpl('pink', N)


def prism(N=256):
    """Prism colormap. MAP = prism(N) returns an Nx3 array."""
    return _cmap_from_mpl('prism', N)


def rainbow(N=256):
    """Rainbow colormap. MAP = rainbow(N) returns an Nx3 array."""
    return _cmap_from_mpl('rainbow', N)


def spring(N=256):
    """Spring colormap. MAP = spring(N) returns an Nx3 array."""
    return _cmap_from_mpl('spring', N)


def summer(N=256):
    """Summer colormap. MAP = summer(N) returns an Nx3 array."""
    return _cmap_from_mpl('summer', N)


def turbo(N=256):
    """Turbo colormap. MAP = turbo(N) returns an Nx3 array."""
    return _cmap_from_mpl('turbo', N)


def viridis(N=256):
    """Viridis colormap. MAP = viridis(N) returns an Nx3 array."""
    return _cmap_from_mpl('viridis', N)


def white(N=256):
    """White colormap (all ones). MAP = white(N) returns an Nx3 array."""
    N = int(N)
    return _fa(np.ones((N, 3), dtype=np.float64))


def winter(N=256):
    """Winter colormap. MAP = winter(N) returns an Nx3 array."""
    return _cmap_from_mpl('winter', N)


# ═══════════════════════════════════════════════════════════════════
# 2. IMAGE I/O
# ═══════════════════════════════════════════════════════════════════

def _require_pil():
    if _PILImage is None:
        raise RuntimeError("Pillow (PIL) is required for image I/O functions")


def imread(filename, *args):
    """Read image from file.

    IMG = imread(FILENAME)
    [IMG, MAP] = imread(FILENAME)  (for indexed images)

    Returns image data as an MxN (grayscale), MxNx3 (RGB), or MxNx4 (RGBA)
    uint8 ForgeArray.
    """
    _require_pil()
    fname = str(_scalar(filename)) if not isinstance(filename, str) else filename
    img = _PILImage.open(fname)

    if img.mode == 'P':
        # Indexed image — return (indices, colormap)
        palette = img.getpalette()  # flat list [R,G,B,R,G,B,...]
        idx_arr = np.array(img, dtype=np.float64)
        if palette is not None:
            cmap = np.array(palette, dtype=np.float64).reshape(-1, 3) / 255.0
        else:
            cmap = _fa(np.empty((0, 3)))
        return _fa(idx_arr), _fa(cmap)

    arr = np.array(img)
    return _fa(arr)


def imwrite(filename, img, *args):
    """Write image to file.

    imwrite(IMG, FILENAME)
    imwrite(IND, MAP, FILENAME)   (indexed image)

    Parameters
    ----------
    filename : str — Output file path.
    img : array — Image data (uint8 or float64 in [0,1]).
    """
    _require_pil()
    fname = str(_scalar(filename)) if not isinstance(filename, str) else filename
    data = _unwrap(img)

    # If float in [0,1], convert to uint8
    if data.dtype.kind == 'f':
        data = np.clip(data * 255, 0, 255).astype(np.uint8)

    if data.ndim == 2:
        pil_img = _PILImage.fromarray(data, mode='L')
    elif data.ndim == 3 and data.shape[2] == 3:
        pil_img = _PILImage.fromarray(data, mode='RGB')
    elif data.ndim == 3 and data.shape[2] == 4:
        pil_img = _PILImage.fromarray(data, mode='RGBA')
    else:
        pil_img = _PILImage.fromarray(data)

    pil_img.save(fname)


def imshow(img, *args):
    """Display image (stub — prints dimensions).

    imshow(IMG)
    imshow(IMG, MAP)

    In non-GUI mode this prints image shape info.
    """
    data = _unwrap(img)
    print(f"imshow: displaying image with shape {data.shape}, dtype {data.dtype}")


def imfinfo(filename):
    """Return image file information as a dict.

    INFO = imfinfo(FILENAME)
    """
    _require_pil()
    fname = str(_scalar(filename)) if not isinstance(filename, str) else filename
    img = _PILImage.open(fname)
    info = {
        'Filename': fname,
        'FileSize': 0,
        'Width': img.width,
        'Height': img.height,
        'BitDepth': 8,
        'ColorType': img.mode,
        'Format': img.format or 'unknown',
        'NumberOfSamples': len(img.getbands()),
    }
    try:
        import os
        info['FileSize'] = os.path.getsize(fname)
    except OSError:
        pass
    return info


def imformats(*args):
    """Return or register image file formats.

    FMTS = imformats()
    Returns a list of supported format strings.
    """
    if _PILImage is None:
        return []
    exts = sorted(_PILImage.registered_extensions().keys())
    return exts


def im2frame(img, *args):
    """Convert image to movie frame struct.

    F = im2frame(IMG)
    F = im2frame(IND, MAP)

    Returns a dict with 'cdata' (the image) and 'colormap'.
    """
    data = _unwrap(img)
    cmap = _unwrap(args[0]) if args else np.empty((0, 3))
    return {'cdata': _fa(data), 'colormap': _fa(cmap)}


def frame2im(frame):
    """Convert movie frame to image.

    IMG = frame2im(F)
    [IND, MAP] = frame2im(F)
    """
    if isinstance(frame, dict):
        cdata = frame.get('cdata', np.array([]))
        cmap = frame.get('colormap', np.empty((0, 3)))
        if np.asarray(cmap).size > 0:
            return _fa(cdata), _fa(cmap)
        return _fa(cdata)
    return _fa(frame)


def getframe(*args):
    """Capture axes or figure as movie frame (stub).

    F = getframe()
    Returns a placeholder frame dict.
    """
    warnings.warn("getframe: GUI capture not available; returning empty frame")
    return {'cdata': _fa(np.zeros((1, 1, 3), dtype=np.uint8)),
            'colormap': _fa(np.empty((0, 3)))}


def movie(*args):
    """Play movie frames (stub).

    movie(M)
    movie(M, N)

    Prints a message — playback not available in headless mode.
    """
    print("movie: playback not available in non-GUI mode")


# ═══════════════════════════════════════════════════════════════════
# 3. COLOR CONVERSION
# ═══════════════════════════════════════════════════════════════════

def hsv2rgb(hsv_in):
    """Convert HSV color values to RGB.

    RGB = hsv2rgb(HSV)
    HSV is an Mx3 or MxNx3 array with H,S,V in [0,1].
    """
    data = _ensure_float(hsv_in)
    if _mpl_colors is not None:
        return _fa(_mpl_colors.hsv_to_rgb(data))

    # Manual conversion
    shape = data.shape
    flat = data.reshape(-1, 3)
    h, s, v = flat[:, 0], flat[:, 1], flat[:, 2]
    i = (h * 6.0).astype(int) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    rgb = np.zeros_like(flat)
    for idx, (r, g, b) in enumerate([(v, t, p), (q, v, p), (p, v, t),
                                      (p, q, v), (t, p, v), (v, p, q)]):
        mask = (i == idx)
        rgb[mask, 0] = r[mask]
        rgb[mask, 1] = g[mask]
        rgb[mask, 2] = b[mask]
    return _fa(rgb.reshape(shape))


def rgb2hsv(rgb_in):
    """Convert RGB color values to HSV.

    HSV = rgb2hsv(RGB)
    RGB is an Mx3 or MxNx3 array with values in [0,1].
    """
    data = _ensure_float(rgb_in)
    if _mpl_colors is not None:
        return _fa(_mpl_colors.rgb_to_hsv(data))

    shape = data.shape
    flat = data.reshape(-1, 3)
    r, g, b = flat[:, 0], flat[:, 1], flat[:, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc
    v = maxc
    s = np.where(maxc > 0, delta / maxc, 0.0)
    h = np.zeros_like(v)
    mask = delta > 0
    rm = mask & (maxc == r)
    gm = mask & (maxc == g)
    bm = mask & (maxc == b)
    h[rm] = ((g[rm] - b[rm]) / delta[rm]) % 6.0
    h[gm] = (b[gm] - r[gm]) / delta[gm] + 2.0
    h[bm] = (r[bm] - g[bm]) / delta[bm] + 4.0
    h = h / 6.0
    h[h < 0] += 1.0
    hsv_arr = np.stack([h, s, v], axis=-1)
    return _fa(hsv_arr.reshape(shape))


def rgb2gray(rgb_in):
    """Convert RGB image to grayscale.

    GRAY = rgb2gray(RGB)
    Uses luminance weights: 0.2989*R + 0.5870*G + 0.1140*B
    """
    data = _ensure_float(rgb_in)
    if data.ndim == 3 and data.shape[2] >= 3:
        gray = 0.2989 * data[:, :, 0] + 0.5870 * data[:, :, 1] + 0.1140 * data[:, :, 2]
    elif data.ndim == 2:
        gray = data
    else:
        gray = data.reshape(-1, 3) @ np.array([0.2989, 0.5870, 0.1140])
    return _fa(gray)


def rgb2ind(rgb_in, n_colors=256):
    """Convert RGB image to indexed image.

    [IND, MAP] = rgb2ind(RGB, N)
    Quantizes to N colors using simple uniform quantization.
    """
    data = _ensure_float(rgb_in)
    n = int(_scalar(n_colors))
    if data.ndim == 3:
        h, w, c = data.shape
        # Simple uniform quantization
        levels = int(round(n ** (1.0 / 3.0)))
        levels = max(levels, 2)
        quantized = np.floor(data * (levels - 1) + 0.5).astype(int)
        quantized = np.clip(quantized, 0, levels - 1)
        ind = quantized[:, :, 0] * levels * levels + quantized[:, :, 1] * levels + quantized[:, :, 2]
        # Build colormap from unique indices
        unique_idx = np.unique(ind)
        cmap = np.zeros((len(unique_idx), 3))
        remap = np.zeros(levels ** 3, dtype=int)
        for i, ui in enumerate(unique_idx):
            r_val = (ui // (levels * levels)) / (levels - 1)
            g_val = ((ui // levels) % levels) / (levels - 1)
            b_val = (ui % levels) / (levels - 1)
            cmap[i] = [r_val, g_val, b_val]
            remap[ui] = i
        ind_remapped = remap[ind]
        return _fa(ind_remapped.astype(np.float64)), _fa(cmap)
    raise ValueError("rgb2ind: input must be an MxNx3 RGB image")


def ind2rgb(ind, cmap):
    """Convert indexed image to RGB using colormap.

    RGB = ind2rgb(IND, MAP)
    """
    idx = _unwrap(ind).astype(int)
    cm = _ensure_float(cmap)
    idx = np.clip(idx, 0, cm.shape[0] - 1)
    rgb = cm[idx]
    return _fa(rgb)


def gray2ind(gray_in, n=256):
    """Convert grayscale image to indexed image.

    [IND, MAP] = gray2ind(GRAY, N)
    """
    data = _ensure_float(gray_in)
    n = int(_scalar(n))
    ind = np.floor(data * (n - 1) + 0.5).astype(int)
    ind = np.clip(ind, 0, n - 1)
    cmap = np.linspace(0, 1, n).reshape(-1, 1) * np.ones((1, 3))
    return _fa(ind.astype(np.float64)), _fa(cmap)


def ind2gray(ind, cmap):
    """Convert indexed image to grayscale.

    GRAY = ind2gray(IND, MAP)
    """
    idx = _unwrap(ind).astype(int)
    cm = _ensure_float(cmap)
    idx = np.clip(idx, 0, cm.shape[0] - 1)
    rgb = cm[idx]
    if rgb.ndim >= 2 and rgb.shape[-1] == 3:
        gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    else:
        gray = rgb
    return _fa(gray)


def im2double(img):
    """Convert image to double precision (float64) in [0,1].

    D = im2double(IMG)
    """
    data = _unwrap(img)
    if data.dtype == np.float64:
        return _fa(data)
    if data.dtype == np.uint8:
        return _fa(data.astype(np.float64) / 255.0)
    if data.dtype == np.uint16:
        return _fa(data.astype(np.float64) / 65535.0)
    if data.dtype.kind == 'f':
        return _fa(data.astype(np.float64))
    # Boolean or other integer
    if data.dtype == bool:
        return _fa(data.astype(np.float64))
    info = np.iinfo(data.dtype)
    return _fa(data.astype(np.float64) / float(info.max))


# ═══════════════════════════════════════════════════════════════════
# 4. COLORMAP OPERATIONS
# ═══════════════════════════════════════════════════════════════════

# Module-level state for current colormap
_current_colormap = None


def colormap(*args):
    """Set or get the current colormap.

    MAP = colormap()         — return current colormap
    colormap(MAP)            — set colormap from Nx3 array
    colormap('name')         — set named colormap
    MAP = colormap('name')   — set and return named colormap
    """
    global _current_colormap

    if len(args) == 0:
        if _current_colormap is None:
            _current_colormap = viridis(256)
        return _current_colormap

    arg = args[0]
    if isinstance(arg, str):
        name_map = {
            'autumn': autumn, 'bone': bone, 'cool': cool, 'copper': copper,
            'cubehelix': cubehelix, 'flag': flag, 'gray': gray, 'hot': hot,
            'hsv': hsv, 'jet': jet, 'lines': lines, 'ocean': ocean,
            'pink': pink, 'prism': prism, 'rainbow': rainbow,
            'spring': spring, 'summer': summer, 'turbo': turbo,
            'viridis': viridis, 'white': white, 'winter': winter,
        }
        fn = name_map.get(arg.lower())
        if fn is None:
            raise ValueError(f"colormap: unknown colormap '{arg}'")
        _current_colormap = fn(256)
    else:
        _current_colormap = _fa(_ensure_float(arg))

    return _current_colormap


def brighten(cmap_in, beta):
    """Brighten or darken a colormap.

    MAP = brighten(MAP, BETA)
    BETA > 0 brightens, BETA < 0 darkens.  |BETA| < 1.
    """
    data = _ensure_float(cmap_in)
    beta = float(_scalar(beta))
    if beta > 0:
        result = data ** (1 - beta)
    elif beta < 0:
        result = data ** (1 / (1 + beta))
    else:
        result = data.copy()
    return _fa(np.clip(result, 0, 1))


def spinmap(*args):
    """Spin the colormap (stub — no GUI).

    spinmap()
    spinmap(T)
    spinmap(T, INC)

    Prints a warning in non-GUI mode.
    """
    warnings.warn("spinmap: colormap animation not available in non-GUI mode")


def rgbplot(cmap_in):
    """Plot the RGB components of a colormap (stub).

    rgbplot(MAP)
    Prints colormap shape info.
    """
    data = _ensure_float(cmap_in)
    print(f"rgbplot: colormap shape {data.shape}, "
          f"R range [{data[:,0].min():.3f}, {data[:,0].max():.3f}], "
          f"G range [{data[:,1].min():.3f}, {data[:,1].max():.3f}], "
          f"B range [{data[:,2].min():.3f}, {data[:,2].max():.3f}]")


def cmpermute(cmap_in, index):
    """Reorder a colormap.

    MAP = cmpermute(MAP, INDEX)
    INDEX is a permutation vector.
    """
    data = _ensure_float(cmap_in)
    idx = _unwrap(index).astype(int).ravel()
    # Convert 1-based to 0-based indexing
    idx = idx - 1
    return _fa(data[idx])


def cmunique(cmap_in):
    """Remove duplicate entries from a colormap.

    MAP = cmunique(MAP)
    """
    data = _ensure_float(cmap_in)
    _, idx = np.unique(data, axis=0, return_index=True)
    idx = np.sort(idx)
    return _fa(data[idx])


def iscolormap(cmap_in):
    """Check if input is a valid colormap.

    TF = iscolormap(MAP)
    A valid colormap is an Nx3 matrix with values in [0,1].
    """
    try:
        data = _ensure_float(cmap_in)
    except Exception:
        return False
    if data.ndim != 2 or data.shape[1] != 3:
        return False
    if np.any(data < 0) or np.any(data > 1):
        return False
    return True


# ═══════════════════════════════════════════════════════════════════
# 5. IMAGE PROCESSING
# ═══════════════════════════════════════════════════════════════════

def image(img, *args):
    """Display image object (stub).

    image(C)
    image(X, Y, C)

    In non-GUI mode, prints shape info.
    """
    if len(args) >= 2:
        c_data = _unwrap(args[1]) if len(args) > 1 else _unwrap(img)
    else:
        c_data = _unwrap(img)
    print(f"image: shape {c_data.shape}, dtype {c_data.dtype}")


def imagesc(img, *args):
    """Display image with scaled colors.

    imagesc(C)
    imagesc(X, Y, C)
    imagesc(..., [CMIN CMAX])

    Delegates to the plotting imagesc for actual display.
    """
    from forge.engine.builtins.plotting import forge_imagesc
    return forge_imagesc(img, *args)


def contrast(img, *args):
    """Adjust image contrast.

    OUT = contrast(IMG)
    OUT = contrast(IMG, LOW, HIGH)

    Stretches intensity range to [0,1] or to [LOW, HIGH].
    """
    data = _ensure_float(img)
    lo = float(_scalar(args[0])) if len(args) > 0 else 0.0
    hi = float(_scalar(args[1])) if len(args) > 1 else 1.0

    dmin, dmax = data.min(), data.max()
    if dmax - dmin == 0:
        return _fa(np.full_like(data, (lo + hi) / 2.0))
    normalized = (data - dmin) / (dmax - dmin)
    result = normalized * (hi - lo) + lo
    return _fa(np.clip(result, lo, hi))


def dither(img, *args):
    """Apply Floyd-Steinberg dithering to convert grayscale to binary.

    BW = dither(GRAY)
    IND = dither(RGB, MAP)

    Parameters
    ----------
    img : array — Grayscale (MxN) or RGB (MxNx3) image, float in [0,1].
    """
    data = _ensure_float(img)

    if data.ndim == 2:
        # Floyd-Steinberg dithering for grayscale -> binary
        h, w = data.shape
        out = data.copy()
        for y in range(h):
            for x in range(w):
                old = out[y, x]
                new = 1.0 if old >= 0.5 else 0.0
                out[y, x] = new
                err = old - new
                if x + 1 < w:
                    out[y, x + 1] += err * 7.0 / 16.0
                if y + 1 < h:
                    if x - 1 >= 0:
                        out[y + 1, x - 1] += err * 3.0 / 16.0
                    out[y + 1, x] += err * 5.0 / 16.0
                    if x + 1 < w:
                        out[y + 1, x + 1] += err * 1.0 / 16.0
        return _fa((out >= 0.5).astype(np.float64))

    if data.ndim == 3:
        # Dither each channel separately, then combine
        gray = rgb2gray(data)
        return dither(gray)

    raise ValueError("dither: input must be 2-D grayscale or 3-D RGB")


# ═══════════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════════


# ── Image binarization ───────────────────────────────────────────

def im2bw(I, threshold=0.5):
    """Convert grayscale image to binary.

    BW = im2bw(I, THRESHOLD)

    Returns a double array of 0s and 1s where pixels > THRESHOLD are 1.
    """
    data = _ensure_float(I)
    t = float(threshold)
    return _fa((data > t).astype(np.float64))


# ── Gaussian filtering ───────────────────────────────────────────

def imgaussfilt(I, sigma=0.5):
    """2-D Gaussian filtering of images.

    B = imgaussfilt(A, SIGMA)

    Applies a Gaussian blur with standard deviation SIGMA.
    """
    from scipy.ndimage import gaussian_filter
    data = _ensure_float(I)
    return _fa(gaussian_filter(data, sigma=float(sigma)))


IMAGE_REGISTRY = {
    # ── Colormaps ──────────────────────────────────────────────────
    'autumn':       autumn,
    'bone':         bone,
    'cool':         cool,
    'copper':       copper,
    'cubehelix':    cubehelix,
    'flag':         flag,
    'gray':         gray,
    'hot':          hot,
    'hsv':          hsv,
    'jet':          jet,
    'lines':        lines,
    'ocean':        ocean,
    'pink':         pink,
    'prism':        prism,
    'rainbow':      rainbow,
    'spring':       spring,
    'summer':       summer,
    'turbo':        turbo,
    'viridis':      viridis,
    'white':        white,
    'winter':       winter,
    # ── Image I/O ─────────────────────────────────────────────────
    'imread':       imread,
    'imwrite':      imwrite,
    'imshow':       imshow,
    'imfinfo':      imfinfo,
    'imformats':    imformats,
    'im2frame':     im2frame,
    'frame2im':     frame2im,
    'getframe':     getframe,
    'movie':        movie,
    # ── Color conversion ──────────────────────────────────────────
    'hsv2rgb':      hsv2rgb,
    'rgb2hsv':      rgb2hsv,
    'rgb2gray':     rgb2gray,
    'rgb2ind':      rgb2ind,
    'ind2rgb':      ind2rgb,
    'gray2ind':     gray2ind,
    'ind2gray':     ind2gray,
    'im2double':    im2double,
    # ── Colormap operations ───────────────────────────────────────
    'colormap':     colormap,
    'brighten':     brighten,
    'spinmap':      spinmap,
    'rgbplot':      rgbplot,
    'cmpermute':    cmpermute,
    'cmunique':     cmunique,
    'iscolormap':   iscolormap,
    # ── Image processing ──────────────────────────────────────────
    'image':        image,
    'imagesc':      imagesc,
    'contrast':     contrast,
    'dither':       dither,
    'im2bw':        im2bw,
    'imgaussfilt':  imgaussfilt,
}

