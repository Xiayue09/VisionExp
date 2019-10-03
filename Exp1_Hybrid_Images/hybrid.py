import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    height = img.shape[0]
    width = img.shape[1]
    new_img = np.zeros((height, width))

    Channel = False
    if img.ndim == 2:
        pass
    else:
        Channel = True
        b, g, r = cv2.split(img)
        new_img_1 = np.zeros((height, width))
        new_img_2 = np.zeros((height, width))
        new_img_3 = np.zeros((height, width))

    identity = kernel.copy()

    id_height = identity.shape[0]
    id_width = identity.shape[1]
    id_go_height = id_height // 2
    id_go_width = id_width // 2

    def xg(a, v):
        sum0 = 0
        for x, y in zip(a.flat, v.flat):
            sum0 += x * y
        return sum0

    def former_img(h, w, id, img):
        new_value_sum = np.zeros(id.shape)
        hh = 0
        for nh in range(h - id_go_height, h + id_go_height + 1):
            ww = 0
            for nw in range(w - id_go_width, w + id_go_width + 1):
                if (nh < 0) or (nh >= height) or (nw < 0) or (nw >= width):
                    new_value_sum[hh][ww] = 0
                else:
                    new_value_sum[hh][ww] = img[nh][nw]
                ww += 1
            hh += 1
        return new_value_sum

    def run_img(identity, im):
        new_image = np.zeros((height, width))
        for h in range(0, height):
            for w in range(0, width):
                v = former_img(h, w, identity, im)
                # print(v)
                value = xg(former_img(h, w, identity, im), identity)
                # print(value)
                new_image[h][w] = value
        return new_image

    if Channel:
        new_img_1 = run_img(identity, b).copy()
        new_img_2 = run_img(identity, g).copy()
        new_img_3 = run_img(identity, r).copy()
        new_img = cv2.merge([new_img_1, new_img_2, new_img_3])
    else:
        new_img = run_img(identity, img)
    return new_img
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    kernel = kernel[::-1,::-1]
    return cross_correlation_2d(img,kernel)
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    g_function = lambda s, x, y: (1 / (2 * np.pi * s ** 2)) * np.e ** (-1 * (x ** 2 + y ** 2) / (2 * s ** 2))
    kernel = np.zeros((height, width))
    go_height = height // 2
    go_width = width // 2

    for h in range(0 - go_height, go_height + 1):
        for w in range(0 - go_width, go_width + 1):
            kernel[h + go_height][w + go_width] = g_function(sigma, w, h)

    change = 1.0 / np.sum(kernel)
    kernel *= change
    return kernel
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    kernel = gaussian_blur_kernel_2d(sigma,size,size)
    return convolve_2d(img,kernel)
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    return img-low_pass(img,sigma,size)
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)


