import copy
import json

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.interpolate as spip
import scipy.ndimage as spim
import skimage.transform as tf
from matplotlib import patches
#from tqdm import tqdm
from typing import Tuple

plt.ion()  # allow blocking plots in the middle of scripts
# savefig
# plt.savefig("filename.pdf", bbox_inches = 'tight', pad_inches = 0)


def bin_image(im, bin_factors=(1, 1)):
    """
    This function will bin an image by the binning bin_factors

    :param im: the image to be binned
    :param bin_factors: the one or two element tuple that will correspond to column, row binning
    :return imbin: binned image
    """

    data_size = im.shape

    if isinstance(bin_factors, int):
        bin_factors = (bin_factors, bin_factors)
    elif len(bin_factors) > 2 or len(bin_factors) < 1:
        raise AssertionError("bin_factors must have 1 to 2 elements.")

    if len(data_size) == 3:
        raise AssertionError("Images must be 2D arrays.")

    if bin_factors != (1, 1):
        j = 0
        for i in bin_factors:
            if i < 1:
                raise ValueError("binning factor cannot be less than 1")
            elif data_size[j] % i != 0:
                raise ValueError(
                    "Binning factor does not evenly divide into both image dimensions. Bin factor = {}, image size = [{}, {}].".format(
                        i, data_size[0], data_size[1]
                    )
                )
            j += 1

    data_size_new = np.true_divide(data_size, bin_factors)
    shape = (data_size_new[0], bin_factors[0], data_size_new[1], bin_factors[1])
    shape = [int(i) for i in shape]
    # it is fair to say I do not understand reshaping in python yet
    return im.reshape(shape).sum(-1).sum(1)


def mask_circ(im_dim, center, radius):
    """
    This function makes a circular mask of size im_dim with a center at center, and radius radius
    """
    y, x = np.ogrid[: im_dim[1], : im_dim[0]]
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = dist <= radius
    # plt.imshow(mask)
    # plt.show()
    return mask


def rot_stack(data, angle):
    """
    This function will take a stack and rotate it counterclockwise by an angle in degrees. It will assume the stack is already centered.
    """
    N = data.shape
    data_rot = np.zeros(N, dtype=data.dtype)
    for i in range(N[2]):
        data_rot[:, :, i] = tf.rotate(data[:, :, i], angle, order=1, mode="wrap")
    return data_rot


def shift_align_stack(
    data: np.ndarray,
    angle_x: float,
    angle_y: float,
    slice_distance: float,
    px_size: float,
) -> Tuple[np.ndarray, float, float]:
    """
    This function will take a 3D numpy array (typically from a ptychography reconstruction) and shift it by angle_x and angle_y to account for residual mistilt

    data: a 3D numpy array arranged (x, y, z), where the beam propagates in the z direction. Alternatively can be seen as (x,y,slice#)
    angle_x: float, mistilt to account for in the x direction. In milliradians
    angle_y: float, mistilt to account for in the y direction. In milliradians
    slice_distance: float, the propagation distance between slices
    px_size: float, the pixel size in the x/y directions in the reconstruction

    Returns:
    shifted_data: 3D numpy array, same dimensions as data
    """

    shifted_data = copy.deepcopy(data)
    slice_distance_px = slice_distance / px_size
    shift_x, shift_y = (
        np.tan(angle_x / 1000) * slice_distance_px,
        np.tan(angle_y / 1000) * slice_distance_px,
    )

    for i in range(data.shape[2]):#had tqdm
        shifted_data[:, :, i] = spim.shift(
            shifted_data[:, :, i], (shift_x * i, shift_y * i), mode="nearest"
        )

    plt.figure(598, clear=True)
    plt.imshow(np.hstack((np.mean(data, axis=2), np.mean(shifted_data, axis=2))))

    return shifted_data, shift_x, shift_y


def remove_plane(im, roi_frac, ang):
    """
    This function will remove an inclined plane background from a reconstruction, similar to the postprocessing step done by Muller, et al. It will only be done in a roi that is fractional from the center of the image. Image will also be rotated
    """
    im = tf.rotate(im, ang, order=1, mode="wrap")
    im_shape = [i for i in im.shape]
    im_sub_shape = [int(roi_frac * i) for i in im_shape]

    im_sub = im[
        np.int((im_shape[0] - im_sub_shape[0]) / 2) : np.int(
            (im_shape[0] + im_sub_shape[0]) / 2
        ),
        np.int((im_shape[1] - im_sub_shape[1]) / 2) : np.int(
            (im_shape[1] + im_sub_shape[1]) / 2
        ),
    ]

    plt.figure(44)
    plt.clf()
    plt.imshow(im_sub)

    # fit plane to illumination
    plt.figure(55)
    plt.clf()
    plt.plot(np.sum(im_sub, axis=0))
    plt.plot(np.sum(im_sub, axis=1))

    xv, yv = np.meshgrid(
        np.arange(im_sub.shape[0]), np.arange(im_sub.shape[1]), indexing="ij"
    )
    basis = np.vstack(
        [np.ndarray.flatten(xv), np.ndarray.flatten(yv), np.ones(xv.size)]
    ).T
    fit = np.linalg.lstsq(basis, np.ndarray.flatten(im_sub), rcond=-1)
    print(fit[0])
    plane = fit[0][0] * xv + fit[0][1] * yv + fit[0][2]
    plane -= np.mean(plane)  # there must be a different way to set mean to 0

    plt.figure(56)
    plt.clf()
    plt.imshow(plane)
    plt.figure(55)
    plt.plot(np.sum(plane, axis=1))
    plt.plot(np.sum(plane, axis=0))
    print(np.mean(plane))

    # subtract plane
    im_flat = im_sub - plane

    plt.figure(57)
    plt.clf()
    plt.imshow(im_flat)

    return im_flat


def com_im(im: np.ndarray, thresh: float =5, plot_flag: bool =False):
    """
    This function will return the center of mass of a greyscale image. Thresh is a value multiplied by the image mean.s
    """
    # test thresholded image
    im_thresh = im > (thresh * np.std(im))
    px = np.where(im_thresh > 0)
    x_cm, y_cm = np.mean(px[0]), np.mean(px[1])

    # plot
    if plot_flag:
        plt.figure(277)
        plt.clf()
        plt.imshow(im_thresh)
        plt.scatter(y_cm, x_cm)
        print("Whole pixel COM is being calculated: {}, {}".format(x_cm, y_cm))
    return x_cm, y_cm


def shift_im_com(
    im: np.ndarray,
    com_target: Tuple[float, float] = None,
    thresh: float = 5,
    order: int = 1,
    plot_flag: bool = False,
):
    """
    This function will shift an image such that the center of mass sits at com_target. It will use some sort of interpolation so that you can do arbitrary coordinate shifts.

    Defaults to first order spline to avoid negative numbers.
    """
    if com_target is None:
        com_target = [i / 2 for i in im.shape]

    com = com_im(im, thresh=thresh)
    shift = np.subtract(com_target, com)
    im_shift = spim.shift(im, shift, order=order, mode="nearest")

    if plot_flag:
        print(f"Target is {com_target}\nOriginal is {com}\nShift is {shift}")
        plt.figure(498)
        plt.clf()
        plt.subplot(121)
        plt.imshow(im)
        circ = patches.Circle(
            (com[1], com[0]), 2, linewidth=2, edgecolor="r", facecolor="none"
        )
        plt.gca().add_patch(circ)
        plt.subplot(122)
        plt.imshow(im_shift)
        circ = patches.Circle(
            (com_target[1], com_target[0]),
            2,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        plt.gca().add_patch(circ)
    return im_shift, shift


def remove_dead_px(data, thresh=0.4):
    """
    This function will find the dead pixels in a stack of images, and replace them with the median value
    """
    N = data.shape
    # find dead pixels
    data_mean = np.mean(data, 2)
    dead_px = find_dead_px(data_mean)
    print(f"{dead_px.shape[0]} pixels fixed.")
    for i in tqdm(range(N[2])):
        for j in dead_px:
            data[j[0], j[1], i] = np.median(
                data[j[0] - 1 : j[0] + 2, j[1] - 1 : j[1] + 2, i]
            )
    return data


def find_dead_px(im, thresh=0.4, plot_flag=False):
    diff_im = np.abs(im - sp.signal.medfilt2d(im))
    dead_px = np.argwhere(diff_im > thresh)
    if plot_flag:
        plt.figure(1)
        plt.clf()
        plt.imshow(im)
        plt.clim(vmax=15)
        ax = plt.gca()
        for i in dead_px:
            ax.add_patch(
                plt.Circle(np.flip(i), radius=2, edgecolor="r", facecolor="None")
            )
    return dead_px


def find_dead_px_median(im, thresh, plot_flag=False):
    """
    This function will return dead px by looking at the ones far below the median values.
    im is the median of the stack
    thresh is the value below which a pixel is dead
    """
    dead_px = np.argwhere(im < thresh)
    if plot_flag:
        plt.figure(2)
        plt.clf()
        plt.imshow(im)
        plt.clim(vmax=150)
        ax = plt.gca()
        for i in dead_px:
            ax.add_patch(
                plt.Circle(np.flip(i), radius=2, edgecolor="r", facecolor="None")
            )

    return dead_px


def click_im_coords(im, xlim=None, ylim=None):
    """
    This will let us click on images, and return an array of where we clicked, in ij, or row column format. This is the typical way to address images, and is not xy format. 
    """
    coords = []

    fig = plt.figure(clear=True)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(im)
    if xlim is not None:
        ax.axes.set_xlim(xlim)
    if ylim is not None:
        ax.axes.set_ylim(ylim)

    count = 0

    def onclick(event):
        nonlocal count
        ix, iy = event.xdata, event.ydata
        print("x = %d, y = %d, type 'y' when finished." % (iy, ix))

        coords.append((iy, ix))  # we flip this here because we want image coordinates
        ax.scatter(ix, iy, c="r")
        ax.annotate(str(count), (ix, iy), c="white")
        plt.draw()

        count += 1

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    while input("Finished? (y/n):\n") != "y":
        print("Continue clicking. ")
    fig.canvas.mpl_disconnect(cid)
    return coords


def get_json_tilts(f):
    """
    This function will read the json file acquired from a tilt tableau and generate the correct xy points.
    The json file is output by exporting the tilt tablea from Nion Swift
    """
    # for compatibility, checks "version" described in json
    versions = [1,13]
    fov = {1:"scan_context_size",13:"scan_context_size"}
    scl = {1:"spatial_calibrations",13:"dimensional_calibrations"}

    # read file
    with open(f, "r") as f_data:
        metadata = f_data.read()

    # parse file
    obj = json.loads(metadata)

    v = obj["version"]
    if v not in versions: # check if still compatible with latest version
        if fov[max(versions)] not in obj.keys():
            raise IndexError("Due to version({}) change of the json_tilts file, the fov parameter is different. Please modify get_json_tilts to add the new naming schema".format(v))
        if scl[max(versions)] not in obj.keys():
            raise IndexError("Due to version({}) change of the json_tilts file, the scl parameter is different. Please modify get_json_tilts to add the new naming schema".format(v))
        v = max(versions) # no errors raised, so ok to use latest version

    fov_x = obj["metadata"]["scan"][fov[v]][0]
    fov_x = int(fov_x)
    scl_x = obj[scl[v]][0]["scale"]
    x = np.linspace(-1, 1, fov_x) * scl_x
    print(x)
    fov_y = obj["metadata"]["scan"][fov[v]][1]
    fov_y = int(fov_y)
    scl_y = obj[scl[v]][1]["scale"]
    y = np.linspace(-1, 1, fov_y) * scl_y

    [x, y] = np.meshgrid(x, y, indexing="ij")
    x = np.reshape(x, fov_x * fov_y)
    y = np.reshape(y, fov_x * fov_y)  # np.ravel(y)

    # Jump through the hoops to unravel
    x = np.ravel(x)
    y = np.ravel(y)

    return x, y


def find_ab(x, y, u, v):
    """
    Find the transformation matrix moving from x and y to u and v. Originally written by Wouter Van den Broek.
    """

    trafo_flag = 2  # A cubic transformation
    no_params = 10  # number of parameters to estimate

    if trafo_flag == -1:
        print("Not enough valid calibration images: QUITING.")
        quit()
    if trafo_flag == 0:
        print("Calculating and correcting a LINEAR distortion model")
    if trafo_flag == 1:
        print("Calculating and correcting a QUADRATIC distortion model")
    if trafo_flag == 2:
        print("Calculating and correcting a CUBIC distortion model")

    A = np.ones((x.shape[0], no_params))
    A[:, 1] = x
    A[:, 2] = y
    if trafo_flag > 0:
        A[:, 3] = x * y
        A[:, 4] = x ** 2
        A[:, 5] = y ** 2
    if trafo_flag > 1:
        A[:, 6] = (x ** 2) * y
        A[:, 7] = x * (y ** 2)
        A[:, 8] = x ** 3
        A[:, 9] = y ** 3

    tmp = np.dot(np.transpose(A), A)
    ab = np.linalg.solve(tmp, np.dot(np.transpose(A), np.transpose([u, v])))

    print(ab)
    return ab


def coordinate_transformation_2d(xy, ab):
    """
    Define the mapping function that will be passed to scipy.ndimage.geometric_transform

    Originally written by Wouter Van den Broek.
    """
    # Transforms from xy to uv
    xy = np.asarray(xy)
    tmp = xy.shape
    if tmp[0] == 2:
        xy = np.transpose(xy)
    tmp = np.asarray(ab.shape)
    if tmp[0] == 3:
        trafo_flag = 0  # linear
    if tmp[0] == 6:
        trafo_flag = 1  # quadratic
    if tmp[0] == 10:
        trafo_flag = 2  # cubic

    x = np.ravel(xy[:, 0])
    y = np.ravel(xy[:, 1])

    a = ab[:, 0]
    u = a[0] + a[1] * x + a[2] * y
    b = ab[:, 1]
    v = b[0] + b[1] * x + b[2] * y
    if trafo_flag > 0:
        u += a[3] * x * y + a[4] * x ** 2 + a[5] * y ** 2
        v += b[3] * x * y + b[4] * x ** 2 + b[5] * y ** 2
    if trafo_flag > 1:
        u += a[6] * x ** 2 * y + a[7] * x * y ** 2 + a[8] * x ** 3 + a[9] * y ** 3
        v += b[6] * x ** 2 * y + b[7] * x * y ** 2 + b[8] * x ** 3 + b[9] * y ** 3

    uv = np.transpose([u, v])
    uv = np.asarray(uv)

    return uv


def coordinate_transformation_2d_areamag(xy, ab):
    """
    Define the local area magnification of the mapping function

    Originally written by Wouter Van den Broek
    """
    # Transforms from xy to uv
    xy = np.asarray(xy)
    tmp = xy.shape
    if tmp[0] == 2:
        xy = np.transpose(xy)
    tmp = np.asarray(ab.shape)
    if tmp[0] == 3:
        trafo_flag = 0  # linear
    if tmp[0] == 6:
        trafo_flag = 1  # quadratic
    if tmp[0] == 10:
        trafo_flag = 2  # cubic

    x = np.ravel(xy[:, 0])
    y = np.ravel(xy[:, 1])

    a = ab[:, 0]
    b = ab[:, 1]

    # Derivative of u wrt x
    duv_dxy = a[1]
    if trafo_flag > 0:
        duv_dxy += a[3] * y + 2 * a[4] * x
    if trafo_flag > 1:
        duv_dxy += 2 * a[6] * x * y + a[7] * y ** 2 + 3 * a[8] * x ** 2
    mag_tmp = duv_dxy

    # Derivative of v wrt y
    duv_dxy = b[2]
    if trafo_flag > 0:
        duv_dxy += b[3] * x + 2 * b[5] * y
    if trafo_flag > 1:
        duv_dxy += b[6] * x ** 2 + 2 * b[7] * x * y + 3 * b[9] * y ** 2
    mag_tmp *= duv_dxy

    # Derivative of u wrt y
    duv_dxy = a[2]
    if trafo_flag > 0:
        duv_dxy += a[3] * x + 2 * a[5] * y
    if trafo_flag > 1:
        duv_dxy += a[6] * x ** 2 + 2 * a[7] * x * y + 3 * a[9] * y ** 2
    mag = duv_dxy

    # Derivative of v wrt x
    duv_dxy = b[1]
    if trafo_flag > 0:
        duv_dxy += b[3] * y + 2 * b[4] * x
    if trafo_flag > 1:
        duv_dxy += 2 * b[6] * x * y + b[7] * y ** 2 + 3 * b[8] * x ** 2
    mag *= duv_dxy

    mag_tmp -= mag
    mag = np.absolute(mag_tmp)
    mag = np.asarray(mag)

    return mag


def unwarp_im(warp_im, ab, method="linear", plot_flag=False, fig_num=900, test=False):
    """
    Originally written by Wouter Van den Broek.
    This takes in a warped image, and a transformation matrix (ab), and outputs a tuple of two images. The first image has the illumination corrected, the second only does the geometric distortions, not the illumination correction
    """
    if test is True:
        warp_im = np.zeros_like(warp_im)
        warp_im[warp_im.shape[0] // 2 :, : warp_im.shape[1] // 2] = 1
        warp_im[: warp_im.shape[0] // 2, warp_im.shape[1] // 2 :] = 2
        warp_im[warp_im.shape[0] // 2 :, warp_im.shape[1] // 2 :] = 3

    # Read the warped image
    g = warp_im.copy()#HACK faster? mantain copy-on-write?(warp_im is never written to) #without the copy, you only have passed the reference, no "reading" has happened. Possible implement this as a parameter?
    offset = 2
    assert np.all(g >= 0), "Values are below 0"
    g = np.log(g + offset)

    # Start the unwarping
    # coordinates of the warped image in uv-space
    u_g = np.arange(g.shape[0])
    v_g = np.arange(g.shape[1])
    uv_g = (u_g, v_g)

    # The mean scaling, probed over many different directions
    sc = np.sqrt(coordinate_transformation_2d_areamag(np.zeros((1, 2)), ab))

    # Coordinates in xy-space
    x_tmp = np.linspace(-0.5, 0.5, g.shape[0]) * (g.shape[0] - 1) / sc
    y_tmp = np.linspace(-0.5, 0.5, g.shape[1]) * (g.shape[1] - 1) / sc
    [x_tmp, y_tmp] = np.meshgrid(x_tmp, y_tmp, indexing="ij")
    x_tmp = np.ravel(x_tmp)
    y_tmp = np.ravel(y_tmp)

    # Transform those to uv-space
    uv_i = coordinate_transformation_2d((x_tmp, y_tmp), ab)

    # Do the unwarping
    g_unwrp = spip.interpn(
        uv_g, g, uv_i, method=method, bounds_error=False, fill_value=np.log(offset)
    )
    g_unwrp = np.reshape(g_unwrp, (g.shape[0], g.shape[1]))

    # undo the logarithms
    if plot_flag: 
        g = np.exp(g) - offset #HACK # only used for plotting, no need to calc...?
    g_unwrp = np.exp(g_unwrp) - offset

    # Correct the uneven illumination

    area_mag = coordinate_transformation_2d_areamag((x_tmp, y_tmp), ab)
    area_mag = np.reshape(area_mag, (g.shape[0], g.shape[1]))
    tmp = area_mag[int(round(g.shape[0] / 2)), int(round(g.shape[1] / 2))]
    area_mag = area_mag / tmp  # Area magnification in the middle is 1 now

    g_unwrp_illmn = g_unwrp * area_mag

    # show the images
    if plot_flag:
        plt.figure(fig_num, clear=True)
        plt.imshow(g)
        plt.title("input")
        plt.figure(fig_num + 1, clear=True)
        plt.imshow(g_unwrp)
        plt.title("unwarped")

        plt.figure(fig_num + 2, clear=True)
        plt.subplot(1, 2, 1)
        plt.title("input")
        plt.imshow(g ** 0.25)
        plt.subplot(1, 2, 2)
        plt.imshow(g_unwrp ** 0.25)
        plt.title("unwarped")

        plt.figure(fig_num + 3, clear=True)
        plt.subplot(121)
        plt.imshow(g_unwrp)
        plt.show()
        plt.title("unwarped")

        plt.subplot(122)
        plt.imshow(g_unwrp_illmn)
        plt.title("illumination corrected")

    return g_unwrp_illmn, g_unwrp

def unwarp_im_old(warp_im, ab, method="linear", plot_flag=False, fig_num=900, test=False):
    """
    Originally written by Wouter Van den Broek.
    This takes in a warped image, and a transformation matrix (ab), and outputs a tuple of two images. The first image has the illumination corrected, the second only does the geometric distortions, not the illumination correction
    """
    if test is True:
        warp_im = np.zeros_like(warp_im)
        warp_im[warp_im.shape[0] // 2 :, : warp_im.shape[1] // 2] = 1
        warp_im[: warp_im.shape[0] // 2, warp_im.shape[1] // 2 :] = 2
        warp_im[warp_im.shape[0] // 2 :, warp_im.shape[1] // 2 :] = 3

    # Read the warped image
    g = warp_im
    offset = 2
    assert np.all(g >= 0), "Values are below 0"
    g = np.log(g + offset)

    # Start the unwarping
    # coordinates of the warped image in uv-space
    u_g = np.arange(g.shape[0])
    v_g = np.arange(g.shape[1])
    uv_g = (u_g, v_g)

    # The mean scaling, probed over many different directions
    sc = np.sqrt(coordinate_transformation_2d_areamag(np.zeros((1, 2)), ab))

    # Coordinates in xy-space
    x_tmp = np.linspace(-0.5, 0.5, g.shape[0]) * (g.shape[0] - 1) / sc
    y_tmp = np.linspace(-0.5, 0.5, g.shape[1]) * (g.shape[1] - 1) / sc
    [x_tmp, y_tmp] = np.meshgrid(x_tmp, y_tmp, indexing="ij")
    x_tmp = np.ravel(x_tmp)
    y_tmp = np.ravel(y_tmp)

    # Transform those to uv-space
    uv_i = coordinate_transformation_2d((x_tmp, y_tmp), ab)

    # Do the unwarping
    g_unwrp = spip.interpn(
        uv_g, g, uv_i, method=method, bounds_error=False, fill_value=np.log(offset)
    )
    g_unwrp = np.reshape(g_unwrp, (g.shape[0], g.shape[1]))

    # undo the logarithms
    g = np.exp(g) - offset
    g_unwrp = np.exp(g_unwrp) - offset

    # Correct the uneven illumination

    area_mag = coordinate_transformation_2d_areamag((x_tmp, y_tmp), ab)
    area_mag = np.reshape(area_mag, (g.shape[0], g.shape[1]))
    tmp = area_mag[int(round(g.shape[0] / 2)), int(round(g.shape[1] / 2))]
    area_mag = area_mag / tmp  # Area magnification in the middle is 1 now

    g_unwrp_illmn = g_unwrp * area_mag

    # show the images
    if plot_flag:
        plt.figure(fig_num, clear=True)
        plt.imshow(g)
        plt.title("input")
        plt.figure(fig_num + 1, clear=True)
        plt.imshow(g_unwrp)
        plt.title("unwarped")

        plt.figure(fig_num + 2, clear=True)
        plt.subplot(1, 2, 1)
        plt.title("input")
        plt.imshow(g ** 0.25)
        plt.subplot(1, 2, 2)
        plt.imshow(g_unwrp ** 0.25)
        plt.title("unwarped")

        plt.figure(fig_num + 3, clear=True)
        plt.subplot(121)
        plt.imshow(g_unwrp)
        plt.show()
        plt.title("unwarped")

        plt.subplot(122)
        plt.imshow(g_unwrp_illmn)
        plt.title("illumination corrected")

    return g_unwrp_illmn, g_unwrp

def prop_plane_wave():
    """
    This function will make and propogate a plane wave to test understanding
    """

    # phase grid
    qx_v = np.fft.fftfreq(256)
    qy_v = np.fft.fftfreq(256)
    qx, qy = np.meshgrid(qx_v, qy_v)
    q2 = qx ** 2 + qy ** 2
    r2 = np.fft.fftshift(q2)
    r2 = np.roll(r2, (30, 50), axis=(0, 1))

    im_real = np.zeros((256, 256))
    im_real[50:75, 123:223] = 1
    im_real[130:180, 50:100] = 0.5
    im_real[r2 < 0.02] = 0.75
    im_wave = im_real
    im_wave_prop = np.fft.fft2(im_wave) * np.exp(-1j * np.pi * q2 * 20)
    im_wave_prop = np.fft.ifft2(im_wave_prop)
    plt.imshow(np.abs(im_wave_prop))

    return


def pad_square(arr, axis=None):
    """
    This function will take in an array and pad it with zeros such that it is square

    axis selects which axes to pad to make square. Will default to (0,1) for 2D arrays and (2,3) for 4D arrays.

    has limited support for N-D arrays. These arrays can only be padded with axis=(0,1), so image axes must be these first two
    """
    if arr.ndim == 4 and axis is None:
        axis = (2, 3)
    elif arr.ndim == 2 and axis is None:
        axis = (0, 1)
    axis = tuple(axis)

    assert axis == (0, 1) or axis == (2, 3), "axis can only be (0,1) or (2,3)"

    new_dim = np.asarray(arr.shape)
    new_dim[axis[0]] = np.max((new_dim[axis[0]], new_dim[axis[1]]))
    new_dim[axis[1]] = np.max((new_dim[axis[0]], new_dim[axis[1]]))
    if axis == (2, 3):
        pad_size = (
            (0, 0),
            (0, 0),
            (
                np.floor((new_dim[2] - arr.shape[2]) / 2).astype(int),
                np.ceil((new_dim[2] - arr.shape[2]) / 2).astype(int),
            ),
            (
                np.floor((new_dim[3] - arr.shape[3]) / 2).astype(int),
                np.ceil((new_dim[3] - arr.shape[3]) / 2).astype(int),
            ),
        )
    elif axis == (0, 1):
        pad_size = [
            (
                np.floor((new_dim[0] - arr.shape[0]) / 2).astype(int),
                np.ceil((new_dim[0] - arr.shape[0]) / 2).astype(int),
            ),
            (
                np.floor((new_dim[1] - arr.shape[1]) / 2).astype(int),
                np.ceil((new_dim[1] - arr.shape[1]) / 2).astype(int),
            ),
        ]
        if arr.ndim != 2:
            for i in range(arr.ndim - 2):
                pad_size.append((0, 0))

    arr = np.pad(arr, pad_size,)
    return arr, pad_size


def fft_analysis(
    im,
    px_size,
    target_resolution,
    window_strength=1,
    fig_start=1,
    clim=None,
    save_fig=False,
):
    """
    This function will simply take in n image, pixel size in real space, and desired resoluion, and will take the fft of the image, appropriately scale it, and put a circle at the desired resolution 

    this currently only works with square images - crop or pad to square to work

    this generates 2 figures
    """

    # plt.close(fig_start)
    # plt.close(fig_start + 1)

    px_size_recip = 1 / (px_size * im.shape[0])
    target_radius = 1 / target_resolution / px_size_recip

    full_window = np.hanning(im.shape[0])[:, None]  # make dot product able
    full_window = np.sqrt(np.dot(full_window, full_window.T)) ** 2

    window_im = full_window * im

    plt.figure(fig_start, clear=True)
    plt.imshow(window_im)
    plt.title("windowed data")

    # now take the fft and put a circle at the right radius - with a ramp to emphasize higher frequency intensities

    xv, yv = np.meshgrid(np.arange(window_im.shape[0]), np.arange(window_im.shape[0]))

    ramp_function = (
        np.sqrt((xv - im.shape[0] / 2) ** 2 + (yv - im.shape[1] / 2) ** 2)
        ** window_strength
    )
    ramp_function /= np.max(ramp_function)
    fft_im = np.abs(np.fft.fftshift(np.fft.fft2(window_im))) ** 0.5
    combined_im = ramp_function * fft_im

    # roughly centered indices for zoomed region
    inds = (
        np.arange(
            (im.shape[0] / 2 - target_radius * 1.2),
            (im.shape[0] / 2 + target_radius * 1.2),
            dtype=int,
        ),
        np.arange(
            (im.shape[0] / 2 - target_radius * 1.2),
            (im.shape[0] / 2 + target_radius * 1.2),
            dtype=int,
        ),
    )
    # ind

    plt.figure(14, clear=True)
    plt.imshow(ramp_function)

    plt.figure(fig_start + 1, clear=True)
    plt.imshow(combined_im, cmap="inferno")
    if clim is None:
        plt.clim(vmin=np.min(combined_im[inds]), vmax=np.max(combined_im[inds]))
    else:
        plt.clim(clim[0], clim[1])
    ax = plt.gca()
    ax.add_patch(
        patches.Circle(
            (im.shape[0] / 2, im.shape[1] / 2),
            target_radius,
            edgecolor="White",
            facecolor="None",
            linestyle=":",
            linewidth=4,
        )
    )
    plt.axis([np.max(inds[0]), np.min(inds[0]), np.min(inds[1]), np.max(inds[1])])
    plt.title(f"circle corresponds to {target_resolution:.2} m")
    plt.show()

    if save_fig:
        plt.axis("off")
        plt.title(None)
        plt.savefig(save_fig, dpi=150, bbox_inches="tight", pad_inches=0)

    return combined_im, full_window


def find_disk(im, com_thresh=100, sigma=6, radius=18, filter=True, plot_flag=False):
    """
    This function finds the center of the brightest disk in an image. 
    com_thresh controls the center of mass cutoff threshhold
    sigma is the gaussian sigma distance,
    radius is the mask radius to exclude the ghosted image
    plot_flag will allow plotting 
    """
    im = copy.deepcopy(im)
    im_m = spim.gaussian_filter(im, sigma=sigma)
    coord = np.where(np.max(im_m) == im_m)
    mask = mask_circ(im.shape, coord[::-1], radius=radius)
    if filter == True:
        im_com = im - im_m  # > np.max(im - im_m) * 0.5
    else:
        im_com = im
    coord = com_im(im_com * mask, thresh=com_thresh)
    max_val = np.max(mask * im)

    if plot_flag:
        plt.figure(10, clear=True)
        plt.imshow(im_com * mask)  # > np.std(im_com*mask)*com_thresh)
        plt.scatter(coord[1], coord[0], color="red")
        try:
            plt.axis(
                [
                    coord[1] - radius * 1.5,
                    coord[1] + radius * 1.5,
                    coord[0] - radius * 1.5,
                    coord[0] + radius * 1.5,
                ]
            )
        except:
            print("bad axes")
            pass
        plt.pause(0.001)

    return coord, max_val


def scale_im(im):
    """
    scale numpy array from 0 to 1
    """
    im_scale = (im - im.min()) / np.ptp(im)

    return im_scale
