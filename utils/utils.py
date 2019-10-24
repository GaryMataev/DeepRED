import matplotlib.pyplot as plt
import numpy as np
import torch
import PIL
from PIL import Image
from skimage.measure import compare_psnr
from utils.data import Data


# ---- Scaling image ---- # 
def pil_resize(pil_img, factor, downscale=True):
    if downscale:
        new_size = [pil_img.size[0] // factor, pil_img.size[1] // factor]
    else:
        new_size = [pil_img.size[0] * factor, pil_img.size[1] * factor]
    new_pil_img = pil_img.resize(new_size, Image.ANTIALIAS)
    return new_pil_img, pil_to_np(new_pil_img)


# ----------- gauss kernel -----------
def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


# -------- Load image and crop it if needed ------ #
def load_and_crop_image(fname, d=1):
    """Make dimensions divisible by `d`"""
    img = Image.open(fname)
    if d == 1: return img, pil_to_np(img)
    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)
    if new_size[0] == img.size[0] and new_size[1] == img.size[1]:
        return img, pil_to_np(img)
    bbox = [
        int((img.size[0] - new_size[0]) / 2),
        int((img.size[1] - new_size[1]) / 2),
        int((img.size[0] + new_size[0]) / 2),
        int((img.size[1] + new_size[1]) / 2),
    ]
    img_cropped = img.crop(bbox)
    return img_cropped, pil_to_np(img_cropped)


# ------- Working with numpy / pil / torch images auxiliary functions -------- #
def save_np(np_img, file, ext='.png'):
    """ saves a numpy image as png (default) """
    pil_img = np_to_pil(np_img)
    pil_img.save(file + ext)


# ---------- compare_psnr ------------
def compare_PSNR(org, est, on_y=False, gray_scale=False):
    assert (on_y==False or gray_scale==False), "Is your image RGB or gray? please choose and try again"
    if on_y:
        return compare_psnr_y(np_to_pil(org), np_to_pil(est))
    if gray_scale:
        return compare_psnr(np.mean(org, axis=0), np.mean(est, axis=0))
    return compare_psnr(org, est)


def load_and_compare_psnr(fclean, fnoisy, crop_factor=1, on_y=False, eng=None):
    # matlab:
    if eng is not None:
        return eng.compare_psnr_y("../" + fclean, "../" + fnoisy, on_y, nargout=1)
    # load:
    _, img_np = load_and_crop_image(fclean, crop_factor)
    _, img_noisy_np = load_and_crop_image(fnoisy, crop_factor)
    # rgba -> rgb
    if img_np.shape[0] == 4: img_np = img_np[:3, :, :]
    if img_noisy_np.shape[0] == 4: img_noisy_np = img_noisy_np[:3, :, :]
    return compare_PSNR(img_np, img_noisy_np, on_y=on_y)


def get_p_signal(im):
    return 10 * np.log10(np.mean(np.square(im)))


def compare_SNR(im_true, im_test):
    return compare_psnr(im_true, im_test, 1) + get_p_signal(im_true)


def rgb2ycbcr(img):
    """
    Image to Y (ycbcr)
    Input:
        PIL IMAGE, in range [0, 255]
    Output:
        Numpy Y Ch. in range [0, 1]
    """
    y = np.array(img, np.float32)
    if len(y.shape) == 3 and y.shape[2] == 3:
        y = np.dot(y, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    return y.round() / 255.0


def rgb2gray(img):
    """
    RGB image to gray scale 
    Input:
        PIL IMAGE, in range [0, 255]
    Output:
        Numpy 3 x Gray Scale in range [0, 1]
        Following the matlab code at: https://www.mathworks.com/help/matlab/ref/rgb2gray.html
        The formula: 0.2989 * R + 0.5870 * G + 0.1140 * B 
    """
    img = np.array(img, np.float32)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = np.dot(img, [0.2989, 0.5870, 0.1140])
    return np.array([img.round() / 255.0]*3, dtype=np.float32)


def compare_psnr_y(org_pil, est_pil):
    return compare_psnr(rgb2ycbcr(org_pil), rgb2ycbcr(est_pil))


# - transformation functions pil <-> numpy <-> torch
def pil_to_np(img_PIL):
    """Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    """
    ar = np.array(img_PIL, np.float32)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar / 255.


def np_to_pil(img_np):
    """Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    """
    ar = np.clip(np.rint(img_np * 255), 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def np_to_torch(img_np):
    """Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    """
    return img_var.detach().cpu().numpy()[0]


def put_in_center(img_np, target_size):
    img_out = np.zeros([3, target_size[0], target_size[1]])

    bbox = [
        int((target_size[0] - img_np.shape[1]) / 2),
        int((target_size[1] - img_np.shape[2]) / 2),
        int((target_size[0] + img_np.shape[1]) / 2),
        int((target_size[1] + img_np.shape[2]) / 2),
    ]

    img_out[:, bbox[0]:bbox[2], bbox[1]:bbox[3]] = img_np

    return img_out


# --------- get noise ---------- #
def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False


def get_noise(input_depth, method, spatial_size, noise_type='u', var=1. / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for filling tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplied by. Basically it is standard deviation scalar.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                           np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False
    return net_input


# ---------- plot functions ----------
def plot_dict(data_dict):
    i, columns = 0, len(data_dict)
    scale = columns * 10  # you can play with it
    plt.figure(figsize=(scale, scale))
    for key, data in data_dict.items():
        i, ax = i + 1, plt.subplot(1, columns, i + 1)
        plt.imshow(np_to_pil(data.img), cmap='gray')
        ax.text(0.5, -0.15, key + (" psnr: %.2f" % (data.psnr) if data.psnr is not None else ""),
                size=36, ha="center", transform=ax.transAxes)
    plt.show()


def matplot_plot_graphs(graphs, x_labels, y_labels):
    total = len(graphs)
    for i, graph in enumerate(graphs):
        plt.figure(figsize=(25, 6))
        ax = plt.subplot(1, total, i + 1)
        plt.plot(graph)
        plt.xlabel(x_labels[i])
        plt.ylabel(y_labels[i], multialignment='center')
    plt.show()


# --------  numpy gray to color -----
def np_gray_to_color(img):
    """ 1 x w x h => 3 x w x h
    """
    img = np.stack([img, img, img], )
    return img


# ------- used for bokeh plots -------
def np_to_rgba(np_img):
    """ ch x w x h => W x H x (ch+1), for alpha
    """
    img = np_img.transpose(1, 2, 0)
    if img.shape[2] == 3:  # 3D image (3, w, h)
        img = 255 * np.dstack([img, np.ones(img.shape[:2])])
    else:  # 2D image (1, w, h)
        img = 255 * np.dstack([img, img, img, np.ones(img.shape[:2])])
    return img.astype(np.uint8)
