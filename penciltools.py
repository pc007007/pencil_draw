import numpy as np
import matplotlib.image as mpimg
from scipy import signal


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


# 将彩色转为灰度
def img2gray(filename):
    img = mpimg.imread(filename)
    gray = rgb2gray(img)
    return gray


# 将灰度图片转为梯度
def gray2gradient(img_gray):
    Alpha_x = np.array([[1, -1], [1, -1]])
    Alpha_y = np.array([[1, 1], [-1, -1]])
    Ix = signal.convolve2d(img_gray, Alpha_x, mode='same')
    Iy = signal.convolve2d(img_gray, Alpha_y, mode='same')
    G = (Ix ** 2 + Iy ** 2) ** 0.5
    return G


# 将直方图拟合
def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def yuv2rgb(im):
    """convert array-like yuv image to rgb colourspace

    a pure numpy implementation since the YCbCr mode in PIL is b0rked.
    TODO: port this stuff to a C extension, using lookup tables for speed
    """
    ## conflicting definitions exist depending on whether you use the full range
    ## of YCbCr or clamp out to the valid range.  see here
    ## http://www.equasys.de/colorconversion.html
    ## http://www.fourcc.org/fccyvrgb.php
    from numpy import dot, ndarray, array
    # if not im.dtype == 'uint8':
    # raise ImageUtilsError('yuv2rgb only implemented for uint8 arrays')

    ## better clip input to the valid range just to be on the safe side
    yuv = ndarray(im.shape)  ## float64
    yuv[:, :, 0] = im[:, :, 0].clip(16, 235).astype(yuv.dtype) - 16
    yuv[:, :, 1:] = im[:, :, 1:].clip(16, 240).astype(yuv.dtype) - 128

    ## ITU-R BT.601 version (SDTV)
    A = array([[1., 0., 0.701],
               [1., -0.886 * 0.114 / 0.587, -0.701 * 0.299 / 0.587],
               [1., 0.886, 0.]])
    A[:, 0] *= 255. / 219.
    A[:, 1:] *= 255. / 112.

    ## ITU-R BT.709 version (HDTV)
    #  A = array([[1.164,     0.,  1.793],
    #             [1.164, -0.213, -0.533],
    #             [1.164,  2.112,     0.]])

    rgb = dot(yuv, A.T)
    return rgb.clip(0, 255).astype('uint8')
