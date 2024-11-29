import numpy as np
import cv2
import os
import config as cfg


def get_amount(obj, bg):
    return cfg.IMAGES_PER_COMBINATION * len(obj) * len(bg)


def get_new_epoch_path():
    m = 0
    existing_dirs = set(os.listdir(cfg.RESULT_DIR))
    while f"epoch{m}" in existing_dirs:
        m += 1
    return os.path.join(cfg.RESULT_DIR, f"epoch{m}/")


def get_objects():
    return [
        cv2.imread(os.path.join(cfg.OBJECTS_DIR, name), -1)
        for name in os.listdir(cfg.OBJECTS_DIR)
        if cfg.IGNORE_FILENAME_SYMBOL not in name
    ]


def put(x, y, obj_img, bg_img):
    bh, bw, c = bg_img.shape
    h, w, a = obj_img.shape
    if 1 - w > x or x > bw - 1 or 1 - h > y or y > bh - 1:
        raise Exception("out of bounds")

    # Compute object cropping boundaries
    fh, th = max(0, -y), min(h, bh - y)
    fw, tw = max(0, -x), min(w, bw - x)

    bg = bg_img.copy()
    paste = bg[max(0, y):min(y + h, bh), max(0, x):min(x + w, bw)]
    obj = obj_img[fh:th, fw:tw, :3]

    bg_alpha = None
    if a == 4:  # If there's an alpha channel
        alpha = obj_img[fh:th, fw:tw, -1] / 255.0
        alpha_n = np.dstack([alpha] * 3)  # Repeat alpha for all channels
        alpha_t = 1.0 - alpha_n
        bg[max(0, y):min(y + h, bh), max(0, x):min(x + w, bw)] = paste * alpha_t + obj * alpha_n
        bg_alpha = np.zeros((bh, bw, c), dtype=np.uint8)
        bg_alpha[max(0, y):min(y + h, bh), max(0, x):min(x + w, bw)] = alpha_n * 255

    else:  # If no alpha channel
        bg[max(0, y):min(y + h, bh), max(0, x):min(x + w, bw)] = obj

    return bg, bg_alpha


def flip_img(img, flip_chance):
    return img if np.random.rand() > flip_chance else cv2.flip(img, 1)


def add_blur(img, blur_chance, blur_spatial_rate, blur_intensity_rate=None):
    blured_img = img
    if np.random.randint(0, 100) > blur_chance * 100: return img
    blur_spatial_rate = get_rate(blur_spatial_rate) * 8
    if blur_intensity_rate:
        blur_intensity_rate = int(get_rate(blur_intensity_rate))
    if blur_spatial_rate == 0: return img
    if cfg.BLUR_TYPE == 'GaussianBlur':
        blured_img = cv2.GaussianBlur(img, (0, 0), blur_spatial_rate)
    elif cfg.BLUR_TYPE == 'bilateralFilter':
        blured_img = cv2.bilateralFilter(img, d=0, sigmaSpace=blur_spatial_rate, sigmaColor=blur_intensity_rate)
    return blured_img


def add_noise(img, noise_rate):
    noise_rate = get_rate(noise_rate)
    arg = int(noise_rate * 127)
    if arg == 0: return img

    if cfg.NOISE_TYPE == 'uniform':
        noise = np.random.randint(-arg, arg, img.shape)
    elif cfg.NOISE_TYPE == 'gaussian':
        noise = np.random.randn(*img.shape) * arg
        noise = np.clip(noise, -arg, arg)

    return np.clip(img, arg, 255 - arg) + noise


def scale_img(img, scale_rate):
    scale_rate = get_rate(scale_rate)
    if scale_rate == 1: return img
    if scale_rate == 0: return np.zeros((1, 1, 4), dtype=np.uint8)

    h, w, _ = img.shape
    return cv2.resize(img, (int(w * scale_rate), int(h * scale_rate)))


def get_rate(rate):
    if isinstance(rate, tuple):
        return np.random.uniform(rate[0], rate[1])
    elif isinstance(rate, list):
        return np.random.choice(rate)
    return rate


def get_coord_info(x, y, h, w, bh, bw):
    cx, cy = (x + w / 2) / bw, (y + h / 2) / bh
    ax, ay = w / bw, h / bh
    return f"{cx} {cy} {ax} {ay}"


def save_as_txt(text, path):
    with open(f"{path}.txt", 'a') as f:
        f.write(text + '\n')


def save_as_grayscale_img(img, path):
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Input image must have shape (height, width, 3)")
    img = np.clip(img, 0, 255).astype(np.uint8)
    if not cv2.imwrite(path, img):
        raise IOError(f"Failed to save image to {path}")


def generate_img(bg_img, obj_img):
    flipped_obj = flip_img(obj_img, cfg.FLIP_PROBABILITY)
    scaled_obj = scale_img(flipped_obj, cfg.OBJECT_SCALE_RANGE)

    # Create blurred image based on alpha channel
    blured_img = scaled_obj.copy()
    alpha = (blured_img[:, :, -1] / 255.0).astype(np.float32)
    blurred_alpha = add_blur(alpha, cfg.BLUR_PROBABILITY, cfg.BLUR_KERNEL_RANGE, cfg.BLUR_INTENSITY_RANGE)
    blured_img[:, :, -1] = (blurred_alpha * alpha * 255).astype(np.uint8)

    oh, ow, c = blured_img.shape
    bh, bw, _ = bg_img.shape
    fx, fy, tx, ty = 0, 0, bw - ow, bh - oh

    if cfg.ALLOW_OUT_OF_BOUNDS:
        out_of_bounds_rate_w = get_rate(cfg.OUT_OF_BOUNDS_RANGE)
        out_of_bounds_rate_h = get_rate(cfg.OUT_OF_BOUNDS_RANGE)
        fx, fy = -int(out_of_bounds_rate_w * ow), -int(out_of_bounds_rate_h * oh)
        tx = bw - int((1 - out_of_bounds_rate_w) * ow)
        ty = bh - int((1 - out_of_bounds_rate_h) * oh)

    if cfg.PLACEMENT_DISTRIBUTION == "gaussian":
        cx, cy = (tx - fx) / 2, (ty - fy) / 2
        x = int(np.random.normal(cx, abs(cx / 1.5 * 8)))
        y = int(np.random.normal(cy, abs(cy / 1.5 * 8)))
        x = np.clip(x, fx, tx)
        y = np.clip(y, fy, ty)
    elif cfg.PLACEMENT_DISTRIBUTION == "uniform":
        x, y = np.random.randint(fx, tx), np.random.randint(fy, ty)
    else:
        x, y = 0, 0

    img, bg_alpha = put(x, y, blured_img, bg_img)
    img_noised = add_noise(img, cfg.NOISE_LEVEL_RANGE)

    if cfg.OUTPUT_FORMAT == 'classification':
        return img_noised, get_coord_info(x, y, oh, ow, bh, bw)
    return img_noised, bg_alpha


