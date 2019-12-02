import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         maxX: int for the maximum X value of a corner
         maxY: int for the maximum Y value of a corner
    """
    #TODO 8
    #TODO-BLOCK-BEGIN
    height = img.shape[0]
    width = img.shape[1]
    dot1 = np.array([[0],[0],[1]])
    dot2 = np.array([[0],[height-1],[1]])
    dot3 = np.array([[width-1],[0],[1]])
    dot4 = np.array([[width-1],[height-1],[1]])
    dot1 = np.matmul(M,dot1)
    dot2 = np.matmul(M,dot2)
    dot3 = np.matmul(M,dot3)
    dot4 = np.matmul(M,dot4)

    x = [dot1[0][0],dot2[0][0],dot3[0][0],dot4[0][0]]
    y = [dot1[1][0],dot2[1][0],dot3[1][0],dot4[1][0]]
    minX = min(x)
    minY = min(y)
    maxY = max(y)
    maxX = max(x)

    # raise Exception("TODO in blend.py not implemented")
    #TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # BEGIN TODO 10
    # Fill in this routine
    #TODO-BLOCK-BEGIN

    M /= M[2][2]
    minX, minY, maxX, maxY = imageBoundingBox(img, M)
    M = np.linalg.inv(M)
    M /= M[2][2]
    height = img.shape[0]
    width = img.shape[1]

    for h in range(minY,maxY):
        for j in range(minX,maxX):
            xy = np.array([[j],[h],[1]])
            xy = np.matmul(M,xy)
            xy /= xy[2][0]
            xx = int(round(xy[0][0]))
            yy = int(round(xy[1][0]))
            if 0 <= xx < width:
                if 0 <= yy < height:
                    if (img[yy][xx]==np.array([0,0,0])).all():
                        continue
                    if 0<width - xx <= blendWidth:
                        addweight = (width-xx)/blendWidth
                    elif xx < blendWidth:
                        addweight = (xx+1) / blendWidth
                    else:
                        addweight = 1
                    acc[h][j][3] += addweight
                    acc[h][j][0] += img[yy][xx][0]*addweight
                    acc[h][j][1] += img[yy][xx][1]*addweight
                    acc[h][j][2] += img[yy][xx][2]*addweight
    #raise Exception("TODO in blend.py not implemented")
    #TODO-BLOCK-END
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    #TODO-BLOCK-BEGIN
    b, g, r, t = cv2.split(acc)
    for h in range(acc.shape[0]):
        for w in range(acc.shape[1]):
            tx = acc[h][w][3]
            if tx != 0:
                b[h][w]/=tx
                g[h][w]/=tx
                r[h][w]/=tx
    img = cv2.merge([b, g, r])
    img = img.astype(np.uint8)
    # raise Exception("TODO in blend.py not implemented")
    #TODO-BLOCK-END
    # END TODO
    return img


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and Returns useful information about the accumulated
       image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all tranformed images lie within acc)
         accHeight: Height of accumulator image(minimum height such that all tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = sys.maxsize
    minY = sys.maxsize
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        #TODO-BLOCK-BEGIN
        minX1, minY1, maxX1, maxY1 = imageBoundingBox(img, M)
        minX = min([minX,minX1])
        minY = min([minY,minY1])
        maxX = max([maxX,maxX1])
        maxY = max([maxY,maxY1])
        #raise Exception("TODO in blend.py not implemented")
        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print('accWidth, accHeight:', (accWidth, accHeight))
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    #TODO-BLOCK-BEGIN
    if is360:
        dis = np.sqrt((x_final-x_init)**2+(y_init-y_final)**2)
        sina = (y_final-y_init)/dis
        cosa = (x_init-x_final)/dis
        A[0][0] = cosa
        A[0][1] = -sina
        A[1][0] = sina
        A[1][1] = cosa
        B = np.identity(3)
        B[0][2] = -width/2
        A = np.matmul(A,B)
    #raise Exception("TODO in blend.py not implemented")
    #TODO-BLOCK-END
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage

