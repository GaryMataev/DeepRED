from utils.utils import *
from models.downsampler import Downsampler


# - blur image - exactly like the NCSR is doing it - 
def get_fft_h(im, blur_type):
    assert blur_type in ['uniform_blur', 'gauss_blur'], "blur_type can be or 'uniform' or 'gauss'"
    ch, h, w = im.shape
    fft_h    =  np.zeros((h,w),)
    if blur_type=='uniform_blur':
        t        =  4 # 9//2
        fft_h[h//2-t:h//2+1+t, w//2-t:w//2+1+t]  = 1/81
        fft_h    = np.fft.fft2(np.fft.fftshift(fft_h))
    else: # gauss_blur
        psf = fspecial_gauss(25, 1.6)
        t = 12 # 25 // 2
        fft_h[h//2-t:h//2+1+t, w//2-t:w//2+1+t]  = psf
        fft_h    =  np.fft.fft2(np.fft.fftshift(fft_h))
    return fft_h


def blur(im, blur_type):
    fft_h = get_fft_h(im, blur_type)
    imout = np.zeros_like(im)
    for i in range(im.shape[0]):
        im_f    =  np.fft.fft2(im[i, :, :])
        z_f     =  fft_h*im_f # .* of matlab
        z       =  np.real(np.fft.ifft2(z_f))
        imout[i, :, :] = z
    return imout


# - the inverse function H - 
def get_h(n_ch, blur_type, use_fourier, dtype):
    assert blur_type in ['uniform_blur', 'gauss_blur'], "blur_type can be or 'uniform' or 'gauss'"
    if not use_fourier:
        return Downsampler(n_ch, 1, blur_type, preserve_size=True).type(dtype)
    return lambda im: torch_blur(im, blur_type, dtype)


def torch_blur(im, blur_type, dtype):
    fft_h = get_fft_h(torch_to_np(im), blur_type)
    fft_h_torch = torch.unsqueeze(torch.from_numpy(np.real(fft_h)).type(dtype), 2)
    fft_h_torch = torch.cat([fft_h_torch, fft_h_torch], 2)
    z = []
    for i in range(im.shape[1]):
        im_torch = torch.unsqueeze(im[0, i, :, :], 2)
        im_torch = torch.cat([im_torch, im_torch], 2)
        im_f    =  torch.fft(im_torch, 2)
        z_f     =  torch.mul(torch.unsqueeze(fft_h_torch, 0), torch.unsqueeze(im_f, 0)) # .* of matlab
        z.append(torch.ifft(z_f, 2))
    z = torch.cat(z, 0)
    return torch.unsqueeze(z[:, :, :, 0], 0)
