import matlab.engine
from utils.utils import *


# ---------- resize using matlab engine --------- #
def matlab_resize(eng, org_img, factor):
    """
    uses matlab imresize
    sometimes it makes a better resized image than python
    to use this faction make a resize.m file in matlab_codes folder
    with the following code:
    '''
    function resize(factor)
        Image_org = imread('tmp.png');
        Image_lr = imresize(Image_org, factor);
        imwrite(Image_lr, 'tmp.png');
    end
    '''
    :param eng: the matlab engine
    :param org_img: the original image (ch, x, y)
    :param factor: the factor to be changed
    :return: numpy image of size(ch, factor*x, factor*y)
    """
    org_img = np_to_pil(org_img)
    org_img.save('matlab_codes/tmp.png')
    eng.resize(float(factor), nargout=0)
    img_pil = Image.open("matlab_codes/tmp.png")
    os.remove("matlab_codes/tmp.png")
    return img_pil, pil_to_np(img_pil)


# ----- Working with Matlab Functions ---- #
# inorder for this code to work properly, make sure you follow the instructions and change the file: 
# larray_sequence.py in the folder PYTHONPATH\Lib\site-packages\matlab\_internal
# https://stackoverflow.com/questions/45284124/improve-performance-of-converting-numpy-array-to-matlab-double/45284125#45284125
def np_to_matlab(np_arr):
    """numpy array -> matlab array"""
    if np_arr.shape[0] == 3:
        return matlab.double(np_arr.transpose(1, 2, 0))
    return matlab.double(np_arr[0, :, :])


# the second function is efficiently convert back
# https://stackoverflow.com/questions/34155829/how-to-efficiently-convert-matlab-engine-arrays-to-numpy-ndarray
def matlab_to_np(mat_arr):
    """matlab array -> numpy array"""
    np_arr = np.array(mat_arr._data, dtype=np.float32).reshape(mat_arr.size, order='F')
    if np_arr.ndim == 2:
        return np.expand_dims(np_arr, axis=0)
    return np_arr.transpose(2, 0, 1)


# --- bm3d_v2 - passing the image as matlab array, import matlab on top to use ----
def bm3d_v2(eng, noisy_np_img, sigma):
    """ this function take around ~45sec. 
    However saving and loading the image take ~12sec x4 faster, 
    so don't use this, use the bm3d function that appear in the notebook
    if you can make it work faster, contact me
    """
    im = np_to_matlab(noisy_np_img)
    if noisy_np_img.shape[0] == 3:  # Color BM3D (3D)
        im = eng.CBM3D_denoise2(im, float(sigma), nargout=1)
        return matlab_to_np(im)
    im = eng.BM3D_denoise2(im, float(sigma), nargout=1)
    return matlab_to_np(im)

