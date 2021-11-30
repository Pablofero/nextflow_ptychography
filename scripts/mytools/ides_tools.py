import glob
import os
import shutil as sh
import struct
import time

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as colors
from matplotlib import patches as patches
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from scipy.special import factorial
#from tqdm import tqdm

matplotlib.rcParams["figure.dpi"] = 100
plt.ion()


def subsample_data(data, scan_shape, step_size, sub_size=None):
    """
    This function will take in a 3D stack of data, shape, and step size, and return a subsampled dataset from the original stack with strides in x and y of size step_size
    """
    N = data.shape
    if np.prod(scan_shape) != N[2]:
        raise AssertionError("Scan shape does not match data shape.")
    if sub_size == None:
        sub_size = scan_shape
    if any([i % step_size != 0 for i in sub_size]):
        print(f"step size = {step_size}")
        print(f"subset size = {sub_size}")
        print("Warning: Step size does not evenly fit into sub size.")

    center = [int(i / 2) for i in scan_shape]
    inds = np.arange(N[2])
    inds = np.reshape(inds, (scan_shape))
    inds_final = inds[
        (center[0] - int(sub_size[0] / 2)) : (
            center[0] + int(sub_size[0] / 2)
        ) : step_size,
        (center[1] - int(sub_size[1] / 2)) : (
            center[1] + int(sub_size[1] / 2)
        ) : step_size,
    ]
    # inds = np.ravel(inds)

    # plots to check
    plt.figure(33)
    plt.clf()
    # im = np.ravel(inds)+1000
    im = np.zeros(len(np.ravel(inds)))
    im[np.ravel(inds_final)] = np.arange(len(np.ravel(inds_final)))

    plt.imshow(np.reshape(im, inds.shape))
    plt.axis("equal")

    data_sub = data[:, :, np.ravel(inds_final)]
    return data_sub


def subsample_beam_positions(
    positions_path, scan_shape, step_size, sub_size=None, write_txt=True
):
    """
    This function will take in a text file of beam positions, and then using the scan shape (tuple), step size (integer), and sub size (tuple) (to be used if you want only a subset around the center of the dataset), generate a new text file of subset beam positions. These positions should correspond to the data created by subsample_data, and so the codes should be used in tandem.

    positions_path will be a path to a file, at which point the file will be searched for the correct "beam position:" syntax
    """
    with open(positions_path) as search:
        skiprows = 0
        max_rows = 0
        for line in search:
            if line[0:14] != "beam_position:" and max_rows == 0:
                skiprows += 1
            elif line[0:14] == "beam_position:":
                max_rows += 1
    print(f"skip rows = {skiprows}, max rows = {max_rows}")

    positions = np.loadtxt(
        positions_path, skiprows=skiprows, max_rows=max_rows, usecols=(1, 2)
    )

    if scan_shape is None and step_size is None:
        return positions

    N = positions.shape
    if np.prod(scan_shape) != N[0]:
        raise AssertionError("Scan shape does not match data shape.")
    if sub_size == None:
        sub_size = scan_shape
    if any([i % step_size != 0 for i in sub_size]):
        print(f"step size = {step_size}")
        print(f"subset size = {sub_size}")
        print("Warning: Step size does not evenly fit into sub size.")

    center = [int(i / 2) for i in scan_shape]
    inds = np.arange(N[0])
    inds = np.reshape(inds, (scan_shape))
    inds_final = inds[
        (center[0] - int(sub_size[0] / 2)) : (
            center[0] + int(sub_size[0] / 2)
        ) : step_size,
        (center[1] - int(sub_size[1] / 2)) : (
            center[1] + int(sub_size[1] / 2)
        ) : step_size,
    ]
    # inds = np.ravel(inds_final)

    positions_sub = positions[np.ravel(inds_final), :]

    if write_txt:
        file = open(os.path.dirname(positions_path) + "/beam_pos.txt", "w")
        for i in positions_sub:
            file.write("beam_position: {:.6e}    {:.6e} \n".format(i[0], i[1]))
        file.close()

    # # plots to check
    plt.figure(33)
    plt.clf()
    scale = 1e9
    plt.scatter(positions[:, 0] * scale, positions[:, 1] * scale, c="b", marker="x")
    plt.scatter(
        positions_sub[:, 0] * scale, positions_sub[:, 1] * scale, c="r", marker="x"
    )
    plt.axis("equal")

    plt.figure(34)
    plt.clf()
    plt.plot(np.ravel(inds_final))

    return positions_sub


def window_diff_data(data, shape, width=2, mean_dp=None):
    """
    This function will take in a 3D datastack data, and pad the 4D data shape with a weighted mean between the mean diffraction pattern and the previous  diffraction patterns.

    The purpose of this function is to see if making the edge of the reconstructed material less "sharp" will decrease Fourier ringing.

    Written by Tom Pekin

    :param data: 3D array
    :param shape: 2 element tuple of scan size
    :param width: int of how many pixels to pad
    :param mean_dp: 2D array the size of 1 of the 3D diffraction patterns, to be averaged with the edge diffraction pattern
    :returns data_window: 3D array now padded with diffraction patterns 
    """

    # Generate the correct indexing
    N = data.shape
    if N[2] != np.prod(shape):
        raise AssertionError("Data shape does not match scan shape")

    if mean_dp == None:
        mean_dp = np.mean(data, 2)

    inds_original = np.reshape(np.arange(N[2]), shape)
    newshape = [shape[i] - width[i] * 2 for i in range(len(shape))]
    print(newshape)
    weights = np.pad(np.ones(newshape), width, "linear_ramp")
    weights = np.sin(np.pi / 2 * weights) ** 2

    data_window = np.zeros((N[0], N[1], np.prod(inds_original.shape)), dtype=data.dtype)

    for i in np.ravel(inds_original):
        if np.ravel(weights)[i] == 1:
            data_window[:, :, i] = data[:, :, i]
        else:
            data_window[:, :, i] = data[:, :, i] * (np.ravel(weights)[i]) + mean_dp * (
                1 - np.ravel(weights)[i]
            )

    # Plot
    plt.figure(15)
    plt.clf()
    plt.imshow(weights)
    ax = plt.gca()
    rect = patches.Rectangle(
        [i - 0.5 for i in width],
        newshape[0],
        newshape[1],
        edgecolor="r",
        facecolor="None",
    )
    ax.add_patch(rect)

    return data_window


def compute_params(
    im, conv_angle=23, acc_voltage=200, thresh=0.5, mn_ratio=3 / 2 * 2 ** 0.5
):
    """
    This will compute the params in the params file for you, based on an image you pass in. This image should be a numpy array for ease of use - can use the output of plot_bin 

    This will print the various params for you

    You can also it with plot_bin(args) instead of im to test image size.

    :param im: numpy array of single CBED used for reconstruction - usually the mean of all the CBEDs
    :param conv_angle: angle in milliradians of the central spot
    :param acc_voltage: accelerating voltage of the microscope in keV
    :param px_size: the desired pixel size in real space, useful for targeting resolutions
    :returns: nothing
    """
    # now we need to compute the pixel sizes for our CBED image using the convergence angle and accelerating voltage
    c = 2.998e8  # speed of light
    m_e = 9.1095e-31  # electron mass
    h = 6.6261e-34  # planck's constant
    e = 1.6022e-19  # charge on electron
    acc_voltage = acc_voltage * 1e3  # convert to volts
    wavelength = h / np.sqrt(
        (2 * m_e * acc_voltage * e * (1 + (e * acc_voltage) / (2 * m_e * c ** 2)))
    )  # in meters

    # determine CBED size in pixels
    diam = 2 * np.sqrt(np.sum(im > thresh * np.mean(im)) / np.pi)
    print(f"diameter is {diam}")
    dK = (2 * conv_angle * 1e-3) / diam * (1 / wavelength)

    print("Make sure these are all whole numbers besides (d1, d2)")
    N = np.asarray(im.shape)
    M = np.ceil(N * mn_ratio)
    # print(M)
    # M = [int(j+1) for j in M if j % 2 != 0]
    M = [int(M[i]) if (M[i] - N[i]) % 2 == 0 else int(M[i] + 1) for i in range(len(M))]
    dn = [(M[i] - N[i]) / 2 for i in range(len(M))]

    # compute pixel sizes
    dR = 1 / dK / M

    # print data
    # print(wavelength)
    print(f"dK reciprocal px size = {dK*1e-9} 1/nm - 1/(dK*CBEDdimX) = real space px size for generate_probe")
    print(f"CBEDDimX, CBEDDimY (n1, n2) = {N} - this is the CBED size")
    print(f"ProbeDimY, ProbeDimX (m1, m2) = {M} - this is a minimum")
    print(f"(dn1, dn2) = {dn} - this should be a wnole number = (m-n)/2")
    print("ObjectDimX, ObjectDimY (l1, l2) must be > ProbeDimX, ProbDimY (m1, m2)")
    print(f"PixelSizeX, PixelSizeY (d1, d2) = {dR[0]:.6e} {dR[1]:.6e} meters")

    # plot images to make sure
    plt.figure(33)
    plt.clf()
    plt.imshow(im / np.max(im))
    rect = patches.Rectangle(
        (
            np.where(im > thresh * np.mean(im))[1].mean() - diam / 2,
            np.where(im > thresh * np.mean(im))[0].mean() + diam / 2,
        ),
        diam,
        -diam,
        facecolor="none",
        edgecolor="r",
    )
    plt.gca().add_patch(rect)
    circ = patches.Circle(
        (
            np.where(im > thresh * np.mean(im))[1].mean(),
            np.where(im > thresh * np.mean(im))[0].mean(),
        ),
        diam / 2,
        facecolor="none",
        edgecolor="y",
        linewidth=2,
    )
    plt.gca().add_patch(circ)
    plt.title(
        "The circle represents the CBED size measured \n If it does not fit, change thresh variable"
    )
    plt.tight_layout()
    plt.show()
    return



def generate_probe(dx, N, voltage, alpha_max, df=0,C3=0,C5=0,C7=0):
    '''
    generates unperturbed probe
    :dx: pixel size in angstrom - 1/(dK*N) * 1e10 is the pixel size from compute_params
    :N: number of pixels on each side
    :voltage: voltage in keV
    :alpha_max: convergence angle in mrad
    :df: defocus in angstrom

    returns:
    :probe: complex initial probe
    '''

    # Parameter
    wavelength = 12.398/np.sqrt((2*511.0+voltage)*voltage) # angstrom

    amax = alpha_max*1e-3 # in rad
    amin = 0

    klimitmax = amax/wavelength
    klimitmin = amin/wavelength
    dk = 1/(dx*N)

    # TODO this is off by 1, max of probe is at 128,128 here, in matlab is 129 129
    kx = np.linspace(-np.floor(N/2), np.ceil(N/2)-1,N)
    # kx = np.fft.fftfreq(N, 1/N)
    [kX,kY] = np.meshgrid(kx,kx)
    kX = kX*dk
    kY = kY*dk
    kR = np.sqrt(kX**2+kY**2)

    mask = (kR<=klimitmax) & (kR>=klimitmin)
    chi = np.pi*wavelength*kR**2*df + np.pi/2*C3*wavelength**3*kR**4 +np.pi/3*C5*wavelength**5*kR**6+np.pi/4*C7*wavelength**7*kR**8 
    phase = np.exp(-1j*chi)
    probe = mask*phase

    probe = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(probe)))
    plot_probe(np.real(probe), np.imag(probe))
    plt.figure(665,clear=True)
    plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(probe))))

    probe = probe/np.sum(np.abs(probe))
    # probe = np.real_if_close(probe)
    return probe
# plot Potential Files


def plot_bin(
    filename,
    size_im=None,
    offset=None,
    fig_num=None,
    save_fig=False,
    no_fig=False,
    clim=(None, None),
):
    """
    This function will plot a binary output file from IDES/ROP and return the array as the specified (or not) shape. 

    inputs:
    filename - path to the .bin file. str
    size_im - size of the binary file to plot in a tuple (row, col), this is the image dimensions. If this is left as None, plot_bin will measure the size of the file in float32s (pixel) and assume a square image with that number of floats. (int, int)
    offset - how many float32s (pixels) to skip before plotting size_im. Requires size_im, and is used typically for moving n*np.prod(size_im) pixels, as this returns the nth image. int
    fig_num - this controls what figure number is used, should be an integer
    save_fig - if True, will save a png file with the same filename as the input.
    no_fig - if True, this will surpress the figure from showing, this is helpful for when you just want the data
    clim - the contrast limits passed into plt.imshow. None uses default. (float, float)

    returns:
    data - a size_im array, or default square array with all the data, as a numpy array. 
    """

    file = open(filename, "rb")
    if offset != None:
        file.seek(offset * 4)
    if offset != None and size_im == None:
        raise AssertionError("When using an offset, size_im must be specified.")
    if size_im == None:
        # assume float32, square images
        file_size = os.fstat(file.fileno()).st_size / 4
        size_im = (int(np.sqrt(file_size)), int(np.sqrt(file_size)))
        # print(file_size)
    data = file.read(np.prod(size_im) * 4)
    data = struct.unpack(np.prod(size_im) * "f", data)
    data = np.asarray(data)
    data = np.reshape(data, size_im)

    if fig_num == None:
        fig_num = 12

    if not no_fig:
        plt.figure(fig_num)
        plt.clf()
        plt.imshow(data, interpolation="none", cmap=plt.cm.gray)
        plt.show()
        file.close()
        plt.clim(clim[0], clim[1])

    if save_fig:
        f_out = filename[:-3] + "png"
        # plt.colorbar()
        plt.axis("off")
        plt.savefig(f_out, bbox_inches="tight", pad_inches=0)

    return data


def plot_bin_slices(filename, size_im, slices=1, gauss_filt=0):
    """
    This function will plot all the slices in a row - will have potential to extend to different shapes of how the slices should be arranged

    filename: a pathname to a valid bin file
    size_im: the size of a single object slice in pixels
    slices: the number of slices to plot. if passed in a tuple, it is (rows, columns), so it will plot row*colum plots
    gauss_filt: a value for the kernel for gaussian filtering. 0 = no filtering
    """
    scale_factor = 0.1
    # plt.close(111)
    f = plt.figure()
    if matplotlib.get_backend() != "MacOSX":
        mgr = plt.get_current_fig_manager()
        mgr.window.setGeometry(
            1980,
            94,
            slices
            * size_im[0]
            * scale_factor
            * 1.05,  # TODO this does not work because slices can now be a tuple
            size_im[1] * scale_factor + 70,
        )  # add 70 for figure menu...

    if isinstance(slices, int):
        slices = (1, slices)

    gs = gridspec.GridSpec(
        nrows=slices[0],
        ncols=slices[1],
        wspace=0.05,
        hspace=0.05,
        figure=f,
        bottom=0,
        top=1,
        left=0,
        right=1,
    )

    ax = [f.add_subplot(gs[i]) for i in range(np.prod(slices))]
    i = 0
    clim = []
    im_all = plot_bin(filename, np.prod(size_im) * np.prod(slices), no_fig=True)

    # link axes so they can zoom together
    for a in ax:
        if i == 0:
            pass
        else:
            ax[0].get_shared_x_axes().join(ax[0], a)
            ax[0].get_shared_y_axes().join(ax[0], a)
        clim = [np.min(im_all), np.max(im_all)]
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.axis("off")
        i += 1
        # a.set_aspect('equal')

    for j, a in enumerate(ax):
        ind_0 = int(np.prod(size_im) * j)
        ind_1 = int(np.prod(size_im) * j + np.prod(size_im))

        im = np.reshape(im_all[ind_0:ind_1], size_im)
        a.imshow(
            gaussian_filter(im, gauss_filt),
            clim=[0.5 * i if gauss_filt > 0 else i for i in clim],
        )

    plt.show()
    return


def gen_beam_positions(scan_size, step_size, rot=0, write_txt=True):
    """
    This function will generate a list of beam positions with an experimentally relevant step size, with 0,0 being the center. It will then rotate them COUNTERCLOCKWISE around the center. Positions are in meters and have the format:
    beam position: x_pos y_pos
    where x_pos and y_pos are numbers
    """
    if type(step_size) == float or type(step_size) == int:
        step_size = [step_size]
        step_size.append(step_size[0])
        # print(step_size)
    x_pos = (
        np.arange(np.ceil(-scan_size[0] / 2), np.ceil(scan_size[0] / 2)) * step_size[0]
    )
    y_pos = (
        np.arange(np.ceil(-scan_size[1] / 2), np.ceil(scan_size[1] / 2)) * step_size[1]
    )

    beam_pos = np.array([[i, j] for j in y_pos for i in x_pos])
    beam_pos_rot = np.zeros(beam_pos.shape)

    theta = np.radians(rot)
    s = np.sin(theta)
    c = np.cos(theta)
    rotMat = np.array(((c, -s), (s, c)))
    for i in range(len(beam_pos)):
        beam_pos_rot[i] = np.matmul(rotMat, beam_pos[i])
    if write_txt:
        file = open("beam_pos.txt", "w")
        for i in beam_pos_rot:
            file.write("beam_position: {:.6e}    {:.6e} \n".format(i[0], i[1]))
        file.close()

    # plot
    # plt.figure(14)
    # plt.clf()
    # scale = 1/np.abs(beam_pos[0,0])
    # # print(scale)
    # plt.scatter([i[0]*scale for i in beam_pos], [i[1]*scale for i in beam_pos], c=np.arange(len(x_pos)*len(y_pos)))
    # plt.axis('equal')
    # plt.scatter([i[0]*scale for i in beam_pos_rot], [i[1]*scale for i in beam_pos_rot], c=np.arange(len(x_pos)*len(y_pos)), cmap=plt.get_cmap('plasma'))
    # plt.axis('equal')
    return beam_pos_rot


def plot_potential_real(
    path_name,
    fname,
    im_size=None,
    offset=None,
    clim=(None, None),
    delay=0.1,
    save_fig=True,
):
    files = glob.glob(path_name + fname)
    files.sort(key=os.path.getmtime)
    for f in files:
        plot_bin(f, im_size, offset, save_fig=save_fig, clim=clim)
        plt.clim(clim[0], clim[1])
        # plt.colorbar()
        plt.title(f)
        # plt.show()
        plt.pause(delay)


def plot_beam_positions(path_name, save_fig=False):
    """
    This function will plot the successive beam positions as IDES iterates. It will sort and plot all of the files in order as given by their date created or name
    """
    files = glob.glob(path_name + "Positions*.txt")
    files.sort(key=os.path.getmtime)

    beam_pos_init = np.loadtxt(files[0], usecols=(1, 2))

    z = np.zeros((beam_pos_init.shape[0], 1))
    beam_pos_init = np.append(beam_pos_init, z, axis=1)
    array_size = list(beam_pos_init.shape) + [len(files)]
    array_size[1] = 4
    beam_pos = np.zeros(array_size)
    beam_pos[:, 0:3, 0] = beam_pos_init
    j = 1
    for f in files[1::]:
        beam_pos_init = np.loadtxt(f, usecols=(1, 2))
        beam_pos[:, 0:2, j] = beam_pos_init
        beam_pos[:, 2, j] = np.sqrt(
            np.sum(np.square(beam_pos[:, 0:2, j] - beam_pos[:, 0:2, j - 1]), 1)
        )
        beam_pos[:, 3, j] = np.sqrt(
            np.sum(np.square(beam_pos[:, 0:2, j] - beam_pos[:, 0:2, 0]), 1)
        )

        j += 1

    # plot the data in a movie format
    plt.figure(12, figsize=(6, 6))
    plt.clf()
    plt.scatter(beam_pos[:, 0, 0], beam_pos[:, 1, 0], c="r", marker="x")
    plot_xmin = np.min(beam_pos[:, 0, 0])
    plot_ymin = np.min(beam_pos[:, 1, 0])
    plot_xmax = np.max(beam_pos[:, 0, 0])
    plot_ymax = np.max(beam_pos[:, 1, 0])
    plt.ylim(plot_ymin, plot_ymax)
    plt.xlim(plot_xmin, plot_xmax)
    plt.title("Initial positions")

    plt.figure(13, figsize=(6, 6))
    plt.clf()
    for i in range(beam_pos.shape[2]):
        plt.figure(13)
        plt.clf()
        plt.scatter(
            beam_pos[:, 0, i],
            beam_pos[:, 1, i],
            c=beam_pos[:, 3, i],
            cmap=plt.cm.plasma,
            marker="x",
            norm=plt.Normalize(0, np.max(beam_pos[:, 3, :])),
        )
        plt.ylim(plot_ymin, plot_ymax)
        plt.xlim(plot_xmin, plot_xmax)
        # plt.title(i+1)
        if save_fig:
            plt.savefig("step" + str(i) + ".png", dpi=300)
        plt.pause(0.5)

    # plt.scatter(beam_pos[:,0,i], beam_pos[:,1,i], c=beam_pos[:,2,i], cmap=plt.cm.plasma, marker='x', norm=plt.Normalize(0,np.max(beam_pos[:,2,:])))
    # plt.ylim(plot_ymin, plot_ymax)
    # plt.xlim(plot_xmin, plot_xmax)
    return beam_pos


def plot_atom_positions(fname, subsample=5):
    """
    This function will plot the atoms in an FDES parameter file to see what the atomic configuration looks like. It will specifically look for lines starting with 'atom:'. The following numbers are Atomic no.; x-, y- and z-coordinate (m); Debeye-Waller factor (m^2); occupancy.
    """
    line_skip = 0
    with open(fname) as f:
        for line in f:
            if line.startswith("atom:"):
                break
            else:
                line_skip += 1
        print(line_skip)

    data = np.loadtxt(fname, skiprows=line_skip, usecols=(2, 3, 4))
    data *= 1e11  # scaling factor
    # plot data
    fig = plt.figure("Atom positions")
    ax = Axes3D(fig)
    ax.scatter(
        data[::subsample, 0], data[::subsample, 1], data[::subsample, 2], marker="."
    )
    return data


def plot_probe(probe_real, probe_imag, mag_phase=False, weighting=.25, save_fig=False):
    """
    This function will plot a probe given a real and imaginary part using HSV, by making the amplitude the V and the phase H.
    If mag_phase is true, it will not compute those, and just plot.
    """
    if not mag_phase:
        probe = probe_real + probe_imag * 1j
        amp = np.absolute(probe)
        ang = np.angle(probe)
    else:
        amp, ang = probe_real, probe_imag

    ang[ang < 0] = ang[ang < 0] + 2 * np.pi
    amp = amp ** weighting

    probesize = [i for i in amp.shape]
    im = np.zeros(ang.size * 3)
    im = np.reshape(im, probesize + [3])
    print(im.shape)
    im[:, :, 2] = amp / np.max(amp)
    im[:, :, 0] = ang / np.max(ang)
    im[:, :, 1] = 1
    im = colors.hsv_to_rgb(im)

    plt.figure(44, clear=True)
    plt.imshow(im)
    plt.axis("off")
    plt.tight_layout()
    if save_fig:
        plt.savefig(
            str(time.time()) + ".png", dpi=150, bbox_inches="tight", pad_inches=0
        )

    return im


def compute_error(data_in, data_out, mask=None, error_metric=1, epsilon=7.4e-5):
    """
    This function will compute the error between the CBED patterns passed in, and the CBED patterns generated by ptyIDES. 
    For error metric 1, this looks like:
    abs(i-j) - [(1.95*(i))**-5 + (0.8*(i)**0.5)**-5]**(-1/5)
    This should converge to 1 at the correct value of mu. 

    For error metric 5 (i+e-j*ln(i+e)), where e is epsilon, in electrons, this looks like:
    (i+e-j*ln(i+e)) - ln(i+e)(i-j)
    and for the second case (error metric 6), the error metric looks the same, but the expectation value calculation is different:
    (i+e-j*ln(i+e)) - [
        i(1-ln(i)) if  i<0.62
        or
        1/2(1+ln(2*pi)) + 1/2*ln(i) if i>=0.62
    ]

    The second case will be called error metric 6.

    data should be in the form of [qx, qy, index]
    mask is a numpy boolean array
    i is data_out from Measurements_model.bin, scaled by a gamma to be in electrons.
    j is data_in, the patterns you feed in, scaled by a gamma to be in electrons.
    """
    if mask is not None:
        data_in = data_in[mask, :]
        data_out = data_out[mask, :]

    if error_metric == 1:
        residual = np.sum(np.abs(data_out - data_in))

        regularization = np.sum(
            ((1.95 * data_out) ** (-5) + (0.8 * (data_out) ** (0.5)) ** (-5)) ** (-0.2)
        )
        error_sum = residual - regularization
    elif error_metric == 5:
        residual = np.sum(data_out + epsilon - data_in * np.log(data_out + epsilon))

        regularization = np.sum(
            data_out + epsilon - data_out * np.log(data_out + epsilon)
        )
        error_sum = residual - regularization
    elif error_metric == 6:
        residual = np.sum(
            data_out
            + epsilon
            - data_in * np.log(data_out + epsilon)
            + np.log(factorial(data_in))
        )

        regularization = np.sum(
            epsilon
            + data_out[data_out < 0.62]
            * (1 - np.log(data_out[data_out < 0.62] + epsilon))
        ) + np.sum(
            0.5 * (1 + np.log(2 * np.pi))
            + epsilon
            + 0.5 * np.log(data_out[data_out >= 0.62])
            + data_out[data_out >= 0.62]
            * np.log(
                data_out[data_out >= 0.62] / (data_out[data_out >= 0.62] + epsilon)
            )
        )

        error_sum = residual - regularization

    return error_sum, residual, regularization


def read_bin_stack(fname, shape):
    """
    This function will read a whole binary file with the size shape. It will assume floats (4 bytes). The last two dimensions of shape should be the CBED size.
    """
    file = open(fname, "rb")
    stack = file.read(np.prod(shape) * 4)
    stack = struct.unpack(np.prod(shape) * "f", stack)
    stack = np.asarray(stack)
    stack = np.reshape(stack, shape)

    return stack


def scale_bin(fname, scale_factor):
    """
    This function will take a .bin file and scale it by the scale factor. It will do it in 1 GB increments such that memory is not a problem. It assumes the data is stored as floats.
    """
    f_read = open(fname, "rb")
    f_write = open(fname[:-4] + "_scaled.bin", "wb")
    fsize = os.path.getsize(fname)
    block_size = int(1e6)
    counter = 0
    while fsize > 0:
        if fsize < 4 * block_size:
            data = np.asarray(struct.unpack(fsize // 4 * "f", f_read.read(fsize)))
            data *= scale_factor
            f_write.write(struct.pack(fsize // 4 * "f", *data))
            fsize -= fsize
        else:
            data = np.asarray(
                struct.unpack(block_size * "f", f_read.read(4 * block_size))
            )
            data *= scale_factor
            fsize -= 4 * block_size
            f_write.write(struct.pack(block_size * "f", *data))
            counter += 1
        if counter % 10 == 0:
            print(f"{fsize/1e9} GB left.")
    f_read.close()
    f_write.close()
    print(f"Finished! {fsize} bytes left.")
    return


def clean_bins(path_name):
    # print(os.listdir(path_name))
    print("Deleted:")
    for f in glob.glob(path_name + "psi*bin"):
        print(f)
        os.remove(f)
    for f in glob.glob(path_name + "Potential*bin"):
        print(f)
        os.remove(f)
    for f in glob.glob(path_name + "Imodel*bin"):
        print(f)
        os.remove(f)
    for f in glob.glob(path_name + "Measurements*bin"):
        print(f)
        os.remove(f)
    for f in glob.glob(path_name + "L1*.bin"):
        print(f)
        os.remove(f)
    for f in glob.glob(path_name + "ItNo.bin"):
        print(f)
        os.remove(f)
    for f in glob.glob(path_name + "ErrorFunction.bin"):
        print(f)
        os.remove(f)
    for f in glob.glob(path_name + "Probe*.bin"):
        print(f)
        os.remove(f)
    for f in glob.glob(path_name + "Object*.bin"):
        print(f)
        os.remove(f)
    for f in glob.glob(path_name + "dE1_*.bin"):
        print(f)
        os.remove(f)

    return


def save_bins(path_name, save_path_name):
    os.makedirs(save_path_name)
    # save PotentialReal files
    print("Copied to {}:".format(save_path_name))
    for f in glob.glob(path_name + "Potential*.bin"):
        sh.copy2(f, save_path_name)
        print(f)
    for f in glob.glob(path_name + "Probe*.bin"):
        sh.copy2(f, save_path_name)
        print(f)
    for f in glob.glob(path_name + "Params*"):
        sh.copy2(f, save_path_name)
        print(f)

    return


def bin_write(data, fname):
    """
    This function will write a ptyIDES compatible bin file.
    data - numpy array, 3D shape (qx,qy,im#)
    fname - file name of saved stack of images, typicall with .bin extension 
    """
    N = data.shape
    file_out = open(fname, "wb")
    data_char = "f"

    if len(N) == 2:
        data = np.ravel(data)
        data = struct.pack(np.prod(N) * data_char, *data)
        file_out.write(data)
    elif len(N) == 3:
        for i in range(N[2]):#tqdm was here
            to_pack = np.ravel(data[:, :, i])
            to_pack = struct.pack(np.prod(N[0:2]) * data_char, *to_pack)
            file_out.write(to_pack)

    file_out.close()

    return None


# run code
plt.ion()
# path_detector = '/media/tom/Data/Ptychography_Simulation/Si_dislo_TDS/img/detector2_4.img'
# path_CBED = '/media/tom/Data/Ptychography_Simulation/Si_dislo_noTDS/img/diffAvg_82_23.img'
# path_top = '/media/tom/Data/Ptychography_Simulation/Si_dislo_noTDS/'

# path_YAO = '/media/tom/Data/Berlin_YAO/20181114_Pty_Merlin/bin/'

# plot_bin(path_top + 'PotentialReal1.bin', (480, 480))
# plot_bin(path_top + 'PotentialReal6.bin', (220, 220))

# compute_params(plot_bin(path_top + 'IntensityIncSum.bin',
#                         (100, 100)), acc_voltage=200)

# plot_bin(path_YAO + 'PotentialReal4.bin', (500,500))
# plot_bin(path_YAO + 'M2-50_YAO_1.bin', (128,128))
# plot_bin(path_YAO + 'PotentialReal1.bin')
# cbed = plot_bin(path_YAO + 'Imodel1_I.bin',(128,128), 128*128*3)

# beams = gen_beam_positions((16,16), 1.4e-10, 103.07)

# # import beam positions from file
# with open('beams_pos.txt') as f:
#     array = np.loadtxt(f)

# data_sub = subsample_data(data_rot, (64,64), 1, (32,32))

# data_pad = window_diff_data(data_sub, (32,32), (7,7))

# plt.figure(13)
# plt.clf()
# plt.imshow(np.reshape(np.mean(np.mean(data_pad,0),0), (32,32)))

# beams = gen_beam_positions((32,32), 1.4e-10, 103.07)
# compute_params(np.mean(data_pad, 2), 25, 300)
# fname = '/media/tom/Data/Berlin_YAO/20181114_Pty_Merlin/bin/storage/M2-50_YAO_3/ptyIDES/mindefocusposrotation_unbinned_1xstep_probeupdated_5multislice_error1_0mu_4pxedge_full_gp/PotentialReal41.bin'
# plot_bin_slices(fname, size_im=(2400,2400), slices=5)

# beam_pos = plot_beam_positions('/mnt/tompekin/code/')
