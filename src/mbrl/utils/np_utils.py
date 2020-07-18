# import cv2
import io
import numpy as np
import PIL, PIL.Image


def imrectify_fisheye(img, K, D, balance=0.0):
    # https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f
    dim = img.shape[:2][::-1]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def imresize(image, shape, resize_method=PIL.Image.LANCZOS):
    assert (len(shape) == 3)
    assert (shape[-1] == 1 or shape[-1] == 3)
    assert (image.shape[0] / image.shape[1] == shape[0] / shape[1]) # maintain aspect ratio
    height, width, channels = shape

    if len(image.shape) > 2 and image.shape[2] == 1:
        image = image[:,:,0]

    im = PIL.Image.fromarray(image)
    im = im.resize((width, height), resize_method)
    im = np.array(im)

    if len(im.shape) == 2:
        im = np.expand_dims(im, 2)

    assert (im.shape == tuple(shape))

    return im


def bytes2im(arrs):
    if len(arrs.shape) == 1:
        return np.array([bytes2im(arr_i) for arr_i in arrs])
    elif len(arrs.shape) == 0:
        return np.array(PIL.Image.open(io.BytesIO(arrs)))
    else:
        raise ValueError


def split_dim_np(np_in, axis, new_shape):
    sh = list(np_in.shape)
    assert axis < len(sh)
    assert sh[axis] == np.prod(new_shape)
    new_shape = sh[:axis] + list(new_shape) + sh[axis + 1:]
    return np_in.reshape(new_shape)


# history is N x hist_len x dim, obs is N x dim
# prepends obs to history along second to last dimension
def advance_history_np(history, obs):
    # print(history.shape)
    if history.shape[-1] == 0:
        return history

    longer = np.concatenate([np.expand_dims(obs, axis=-2), history], axis=-2)
    return longer[:, :-1]  # delete the last element
