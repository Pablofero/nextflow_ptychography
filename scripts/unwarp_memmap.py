import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from math import ceil
import os

# setup import of own tools.
# append the path to the folder containing im_tools.py and __init__.py
# sys.path.append("/Users/Tom/Documents/Research/code/pyscripts/") #.#
#sys.path.append(os.path.join(sys.path[0],'scripts'))
print("sys.version is", sys.version)
#print(f'{sys.path[0]=}')
from mytools import im_tools
################################### -mod- ############################################################
################################### UNFINISHED ############################################################

# for config import, commandline and json/yaml file
from jsonargparse import ArgumentParser, ActionConfigFile
# for defining custom file_path type and geting allredy define ones
from jsonargparse.typing import path_type, Path_fr, Path_dw
# for specifying type of input, checks things before anything is actually needed https://jsonargparse.readthedocs.io/en/stable/index.html#type-hints
from typing import List
# define a custom path type for testing, use this and other: https://jsonargparse.readthedocs.io/en/stable/#jsonargparse.typing.Path_fr
Path_nocheck = Path_dw # TODO path_type('rw', docstring='str pointing to a file', skip_check=True) #
Path_f_nocheck =  Path_fr#path_type('frw', docstring='str pointing to a file', skip_check=True)#
# Path_fr->file that exists and is readable
# Path_fc->that can be created if it does not exist
# Path_dw->directory exists and is writeable
# Path_dc->directory that can be created if it does not exist
# Path_drw->directory that exists and is readable and writeable

# help= can be used in the future to autogenerate yaml files with comments (https://jsonargparse.readthedocs.io/en/stable/index.html#configuration-files)
parser = ArgumentParser(parse_as_dict=True)
parser.add_argument('--cfg', action=ActionConfigFile)
parser.add_argument("--cpu_count", type=int, help="amount of cores to be used", required=True)
# more about how to use: https://jsonargparse.readthedocs.io/en/stable/#parsing-paths
parser.add_argument("--stem_data", type=Path_nocheck, help="root folders where all the files are", required=True)
parser.add_argument("--stem_results", type=Path_nocheck, help="root folders where all the files are", required=True)
parser.add_argument("--f_ref", type=Path_f_nocheck, help="tilt tableau datasets, ronchicam, 4D stack", required=True)
parser.add_argument("--f_warp", type=Path_f_nocheck, help="tilt tableau datasets, dectris, 4D stack", required=True)
parser.add_argument("--f_list", type=List[Path_f_nocheck], help="raw data stack path - 4D numpy files extracted from Swift", required=True)
parser.add_argument("--find_points_ref", type=bool, help="if false, the code expects a text file of beam positions in the format 'x_pos y_pos', with each location a new line", default=True)
parser.add_argument("--find_points_warp", type=bool, help="if false, the code expects a text file of beam positions in the format 'x_pos y_pos', with each location a new line", default=True)
parser.add_argument("--dist_min", type=int, help="how far apart (in pixels) do the detected disks in the ronchigram stack need to be, to be considered a valid disk", default=20)
parser.add_argument("--plot_dist_min_flag", type=bool, help="look at the histogram to figure out the correct number", default=False)
parser.add_argument("--plot_flag", type=bool, help="this is slow, so return to false once the parameters above and below are found.", default=False)
parser.add_argument("--radius_ref", type=int, help="the radius used in im_tools.mask_circ. 15 good for 100mrad_pelz, 30 good for DiffRef", default=15)
parser.add_argument("--radius_warp", type=int, help="the radius used in im_tools.mask_circ", default=20)
parser.add_argument("--thresh_ref", type=int, help="threshhold value for im_tools.com_im", default=45)
parser.add_argument("--thresh_warp", type=int, help="threshhold value for im_tools.com_im", default=4)
parser.add_argument("--use_json", type=bool, help="this is currently in beta and not to be used without explanation from Tom, in which order the data should be processsed. These require variables defined from find_points_ref and find_points_warp", default=False)
parser.add_argument("--f_json", type=Path_f_nocheck, help="raw data stack path - 4D numpy files extracted from Swift", required=False)
parser.add_argument("--save_all", type=bool, help="save dewarping parameters once all are successfully found as text files in the folder", default=False)
parser.add_argument("--load_data", type=bool, help="If you have everything above working, and saved the necessary text files, set everything above to False and everything below to True, and you will save a new, unwarped, 4D stack", default=False)
parser.add_argument("--process_data", type=bool, help="see above", default=False)
parser.add_argument("--save_data", type=bool, help="see above", default=False)


#####################################  legacy comments ################################################
#######################################################################################################

# plt.ion()

# The master unwarp.py file is found in /Users/Tom/Documents/Research/code/
# pyscripts/. It is copied into research specific folders so that parameters
# can be changed. ###

# root folders where all the files are
stem_data = Path(
    "/Users/Tom/Documents/Research/Data/2021/2021-02-16_bilayer_graphene_ptychography/exports"
)
stem_results = Path(
    "/Users/Tom/Documents/Research/Results/HU Postdoc/2021-02-16_bilayer_graphene_ptychography/dewarping_results/100mrad_pelz"
)

# tilt tableau datasets - ref is ronchicam, warp is dectris. these are 4D stacks
f_ref = stem_data / "Spectrum Image (RonchiCam)_100mrad_pelz_tiltmap.npy"
f_warp = stem_data / "Spectrum Image (Dectris)_100mrad_pelz_tiltmap.npy"
# f_ref = stem_data / "Spectrum Image (RonchiCam))_DiffRef_tiltmap.npy"
# f_warp = stem_data / "Spectrum Image (Dectris)_DiffRef_tiltmap.npy"

# raw data stack path - 4D numpy files extracted from Swift. Keep in mind
# stem_data is the base path to the folder defined above.

f_list = [
    Path(
        "/Users/Tom/Documents/Research/Data/2021/2021-02-16_bilayer_graphene_ptychography_second_library/exports/Spectrum Image (Dectris)_100mrad_pelz_filtered_21.13pm.npy"
    ),
    Path(
        "/Users/Tom/Documents/Research/Data/2021/2021-02-16_bilayer_graphene_ptychography_second_library/exports/Spectrum Image (Dectris)_100mrad_pelz_unfiltered_21.25pm.npy"
    ),
]


# f_list = [
#     Path(
#         "/Users/Tom/Documents/Research/Data/2021/2021-02-16_bilayer_graphene_ptychography_second_library/exports/Spectrum Image (Dectris)_DiffRef_filtered_21.03pm.npy"
#     ),
#     Path(
#         "/Users/Tom/Documents/Research/Data/2021/2021-02-16_bilayer_graphene_ptychography_second_library/exports/Spectrum Image (Dectris)_Diff_ref_unfiltered_20.58pm.npy"
#     ),
# ]


# control flow and user defined parameters
# first you find the beam in the reference 4D stack, and then the warped stack
# fit the following flags are set to false, the code expects a text file of beam
# positions in the format 'x_pos y_pos', with each location a new line
find_points_ref = True
find_points_warp = True
# how far apart (in pixels) do the detected disks in the ronchigram stack need to be, to be considered a valid disk
dist_min = 20


# if this does not seem to be working, first see what is going on with the
# following flag set to True. This is slow, so return to false once the
# parameters above and below are found.
plot_flag = False

# there are two more parameters in the code below, the radius used in
# im_tools.mask_circ and the threshhold value for im_tools.com_im. The default
# values should work for standard data
# radius_ref = 30 # good for DiffRef
radius_ref = 15  # good for 100mrad_pelz
radius_warp = 10
thresh_ref, thresh_warp = 45, 4

# this is currently in beta and not to be used without explanation from Tom, in which order the data should be processsed. These require variables defined from find_points_ref and find_points_warp
use_json = True
# f_json = Path('/Users/Tom/Documents/Research/Data/2021/2021-02-16_bilayer_graphene_ptychography/exports/Spectrum Image (Dectris)_DiffRef_tiltmap.json')
f_json = Path('/Users/Tom/Documents/Research/Data/2021/2021-02-16_bilayer_graphene_ptychography/exports/Spectrum Image (Dectris)_100mrad_pelz_tiltmap.json')
#  x, y = im_tools.get_json_tilts(f_json) -mod- ############################################################

# save dewarping parameters once all are successfully found as text files in the folder
save_all = False

# work on and save raw 4D stacks - this is now working on the experimental data
# for ptychography, etc. If you have everything above working, and saved the
# necessary text files, set everything above to False and everything below to
# True, and you will save a new, unwarped, 4D stack
load_data = False  # True
process_data = False  # True
save_data = False


#####################################  end legacy comments ############################################
#######################################################################################################

params = parser.parse_args()

# TODO Nextflow? get usable processors: len(os.sched_getaffinity(0))
cpu_count = params["cpu_count"]
stem_data = Path(params["stem_data"])
stem_results = Path(params["stem_results"])
f_ref = stem_data / params["f_ref"]
f_warp = stem_data / params["f_warp"]
f_list = [Path(p) for p in params["f_list"]]  # extrac from list
find_points_ref = params["find_points_ref"]
find_points_warp = params["find_points_warp"]
dist_min = params["dist_min"]
plot_dist_min_flag = params["plot_dist_min_flag"]
plot_flag = params["plot_flag"]
radius_ref = params["radius_ref"]
radius_warp = params["radius_warp"]
thresh_ref = params["thresh_ref"]
thresh_warp = params["thresh_warp"]
use_json = params["use_json"]
f_json = Path(params["f_json"])
save_all = params["save_all"]
load_data = params["load_data"]
process_data = params["process_data"]
save_data = params["save_data"]

#######################################################################################################
#######################################################################################################
x, y = im_tools.get_json_tilts(f_json)
### no more user defined inputs below this point ###


ref_im = np.load(f_ref, mmap_mode="r")
warp_im = np.load(f_warp)

# make Dectris data square, to prevent clipping
warp_im, pad_size = im_tools.pad_square(warp_im, axis=(2, 3))
# warp_im /= np.max(warp_im)
# ref_im /= np.max(ref_im)
# ref_im -= np.min(ref_im)

# change shape of ref_im from (x, y, x',y') to (i, x', y') witha view, so no actula data is moved around

if find_points_ref:
    print("processing find_points_ref...", end='')
    ref_imV = ref_im.view()  # check that nothing is copied! (it errors out in the next line)
    ref_imV.shape = (ref_im.shape[0]*ref_im.shape[1],
                     ref_im.shape[2], ref_im.shape[3])
    coords, max_vals = np.empty(
        (ref_imV.shape[0], 2)), np.empty(ref_imV.shape[0])
    def find_disk_ref_im(i):
        # speed up parallelization with copy-on-write (forking, not windows)
        global ref_imV
        return im_tools.find_disk(
            ref_imV[i],
            com_thresh=thresh_ref,
            radius=radius_ref,
            filter=True,
            plot_flag=plot_flag,
            sigma=8,  # good for 100mrad_pelz. Used 16 for DiffRef #TODO extra parameter
        )
    with mp.Pool(cpu_count) as p:
        # imap(find_disk,ref_imV,ceil(ref_imV.shape[0]/p._processes)) for less memory
        res = p.map(find_disk_ref_im, range(
            ref_imV.shape[0]), ceil(ref_imV.shape[0]/cpu_count))
    for i, res_iter in zip(range(ref_imV.shape[0]), res):
        coords[i] = [res_iter[0][0], res_iter[0][1]]
        max_vals[i] = res_iter[1]
    print("done")
else:
    print(
        f"Loading data from {str(stem_results)}/coords.txt and /max_vals.txt. This will fail if these do not exist."
    )
    coords = np.loadtxt(stem_results / "coords.txt")
    max_vals = np.loadtxt(stem_results / "max_vals.txt")

if find_points_warp:
    print("processing find_points_warp...", end='')
    # check that nothing is copied! (it errors out in the next line)
    warp_imV = warp_im.view()
    warp_imV.shape = (
        warp_im.shape[0]*warp_im.shape[1], warp_im.shape[2], warp_im.shape[3])
    coords_warp, max_vals_warp = np.empty(
        (warp_imV.shape[0], 2)), np.empty(warp_imV.shape[0])
    def find_disk_warp_im(i):
        # speed up parallelization with copy-on-write (forking, not windows)
        global warp_imV
        return im_tools.find_disk(
            warp_imV[i],
            com_thresh=thresh_warp,
            radius=radius_warp,
            plot_flag=plot_flag,
        )
    with mp.Pool(cpu_count) as p:
        # imap(find_disk,warp_imV,ceil(warp_imV.shape[0]/p._processes)) for less memory
        res = p.map(find_disk_warp_im, range(
            warp_imV.shape[0]), ceil(warp_imV.shape[0]/cpu_count))
    for i, res_iter in zip(range(warp_imV.shape[0]), res):
        coords_warp[i] = [res_iter[0][0], res_iter[0][1]]
        max_vals_warp[i] = res_iter[1]
    print("done")
else:
    print(
        f"Loading data from {str(stem_results)}/coords_warp.txt and /max_vals_warp.txt. This will fail if these do not exist."
    )
    coords_warp = np.loadtxt(stem_results / "coords_warp.txt")
    max_vals_warp = np.loadtxt(stem_results / "max_vals_warp.txt")

# remove bad images by various metrics.
inds = np.ones(max_vals_warp.shape[0]).astype(bool)
inds[max_vals_warp < 0.5 * np.mean(max_vals_warp)] = False
inds[max_vals < np.mean(max_vals)] = False

# minimum distance metric
# set a minimum distance between spots due to ronchiogram ghosting
dist = np.zeros_like(inds).astype(float)
for i in range(len(inds) - 1):
    dist[i + 1] = np.linalg.norm(coords[i + 1, :] - coords[i, :])

# look at the histogram to figure out the correct number
if plot_dist_min_flag:
    plt.hist(dist[inds], 100)
inds[dist < dist_min] = False
# inds[dist>1000] = False

# save values from finding disks as text files
os.makedirs(stem_results, exist_ok=True)
if save_all:
    np.savetxt(stem_results / "coords.txt", coords)
    np.savetxt(stem_results / "max_vals.txt", max_vals)
    np.savetxt(stem_results / "coords_warp.txt", coords_warp)
    np.savetxt(stem_results / "max_vals_warp.txt", max_vals_warp)
    np.savetxt(stem_results / "inds.txt", inds)
    np.savetxt(stem_results / "dist.txt", dist)

# fit surface, ab is the distortion matrix
warp_avg = np.mean(warp_im, axis=(0, 1))
ab = im_tools.find_ab(
    coords[inds, 0] - np.mean(coords[inds, 0]),
    coords[inds, 1] - np.mean(coords[inds, 1]),
    coords_warp[inds, 0],
    coords_warp[inds, 1],
)

if use_json:
    ab = im_tools.find_ab(x[inds] - np.mean(x[inds]), y[inds] -
                          np.mean(y[inds]), coords_warp[inds, 0], coords_warp[inds, 1])

# this command unwarps the image
im_tools.unwarp_im(warp_avg, ab, plot_flag=plot_flag)
if plot_flag:
    plt.pause(5)  # HACK it was plt.pause(0.01), it disapears in my machine

# maybe still need to fix gain? using original image?  i don't think so

# now we unwarp the 4D stacks of actual data
for f in f_list:
    if load_data:
        data = np.load(f, mmap_mode="r")
        print(data.shape)
        if process_data:
            # print("padding real data...") #HACK no needed?: pad_size = ((0, 0), (0, 0), (0, 0), (0, 0))
            # data = np.pad(
            #     data, pad_size
            # )  # this is perhaps unnecessary now that my padding function can handle 2D images. This step takes a lot of memory
            # data_mean = np.mean(data, axis=(0, 1)) #HACK this is not used?!

            print('reserving shared memory')
            dataV = data.view()
            # print(dataV.shape)
            dataV.shape = (data.shape[0]*data.shape[1], data.shape[2], data.shape[3])# data is ijkm, dataV is ĩkm
            print('creating RawArray...',end='')
            data_out_ctypes_arr = mp.RawArray(np.ctypeslib.as_ctypes_type(dataV.dtype),dataV.size)# Note that setting and getting an element is potentially non-atomic, ok in this case though as each process reads/writes data orthogonally to the others!
            print('done')
            print('creating np.array...')
            data_out_1d = np.ctypeslib.as_array(data_out_ctypes_arr,dataV.shape)
            data_outV = data_out_1d.view()
            data_outV.shape = (data.shape[0]*data.shape[1], data.shape[2], data.shape[3])
            # print(data_outV.shape)

            # data_outV.shape = (data_out.shape[0]*data_out.shape[1],
            #             data_out.shape[2], data_out.shape[3])
            print("Unwarping the 4D stacks of the actual data...", end='')
            def unwarp_im_data(i):
                #print(i)
                global dataV #we are modifying a global variable
                data_outV[i] = im_tools.unwarp_im(#################HACK modified to not write dataV ->~10s less for ~10gb
                    dataV[i], ab, plot_flag=False)[0]
            with mp.Pool(cpu_count) as p:
                # imap(find_disk,dataV,ceil(dataV.shape[0]/p._processes)) for less memory
                #-#res = 
                p.map(unwarp_im_data, range(dataV.shape[0]))#, ceil(dataV.shape[0]/cpu_count)
                #p.map(unwarp_im_data, range(100), ceil(dataV.shape[0]/cpu_count))
                #p.map(unwarp_im_data, range(100), ceil(100/cpu_count))
            
            # data_out = np.asarray(res)
            # before = data[0].copy()
            data_out = data_out_1d.view()
            data_out.shape = (data.shape[0],data.shape[1],
                        data.shape[2], data.shape[3])
            # data = np.asarray(res)
            print("done")
            #print("before:\n",data[0])
            #print("after:\n",data_out[0])
            # print("as res:\n",data_out_1d[0])
            print("calculating mean...",end='')
            data_unwarp_mean = np.mean(data_out, axis=(0, 1))
            print("done")
            #-# data_unwarp_mean = np.mean(data, axis=(0, 1))

            plt.figure(34, clear=True)
            plt.imshow(data_unwarp_mean ** 0.25)

            if save_data:
                #data_out.flush()
                f_out = f.parent / (f.stem + "_unwarped.npy")
                if use_json:
                    f_out = f.parent / (f.stem + "_unwarped_json.npy")
                np.save(f_out, data_out)
                #-# np.save(f_out, data)
                print(f"saved to {f_out}")
            
