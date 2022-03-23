from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from math import ceil
import os
import sys


# print(os.getcwd())
# print('sys.path[0]',f'{os.path.split(sys.path[0])[0]}')
# setup import of own tools.
# append the path to the folder containing im_tools.py and __init__.py
sys.path.append(os.path.split(sys.path[0])[0]) # /scripts
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
Path_nocheck = path_type('rw', docstring='str pointing to a file', skip_check=True) #
Path_f_nocheck =  path_type('frw', docstring='str pointing to a file', skip_check=True)#
# Path_fr->file that exists and is readable
# Path_fc->that can be created if it does not exist
# Path_dw->directory exists and is writeable
# Path_dc->directory that can be created if it does not exist
# Path_drw->directory that exists and is readable and writeable

# help= can be used in the future to autogenerate yaml files with comments (https://jsonargparse.readthedocs.io/en/stable/index.html#configuration-files)
parser = ArgumentParser(parse_as_dict=True)
parser.add_argument('--cfg', action=ActionConfigFile)
# more about how to use: https://jsonargparse.readthedocs.io/en/stable/#parsing-paths
parser.add_argument("--unwarp_ref", type=Path_f_nocheck, help="tilt tableau datasets, ronchicam, 4D stack", required=True)
parser.add_argument("--unwarp_warp", type=Path_f_nocheck, help="tilt tableau datasets, dectris, 4D stack", required=True)
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
parser.add_argument("--precomputed_json", type=Path_f_nocheck, help="raw data stack path - 4D numpy files extracted from Swift", required=False)
parser.add_argument("--save_all_precompute", type=bool, help="save dewarping parameters once all are successfully found as text files in the folder", default=False)
parser.add_argument("--cpu_count", type=int, help="amount of cores to be used", required=True)
# parser.add_argument("--stem_results", type=Path_nocheck, help="folders where all precomputed results for the ab matrix is stored, see --save_all_precompute", required=False)

params = parser.parse_args()

cpu_count = params["cpu_count"]
f_ref = Path(params["unwarp_ref"])
f_warp =  Path(params["unwarp_warp"])
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
f_json = Path(params["precomputed_json"])
save_all_precompute = params["save_all_precompute"]
#######################################################################################################
#######################################################################################################
if use_json:
    x, y = im_tools.get_json_tilts(f_json)


ref_im = np.load(f_ref, mmap_mode="r")
warp_im = np.load(f_warp)

# make Dectris data square, to prevent clipping
warp_im, pad_size = im_tools.pad_square(warp_im, axis=(2, 3))
# warp_im /= np.max(warp_im)
# ref_im /= np.max(ref_im)
# ref_im -= np.min(ref_im)

# change shape of ref_im from (x, y, x',y') to (i, x', y') with a view, so no actual data is moved around

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
            ref_imV.shape[0]))#, ceil(ref_imV.shape[0]/cpu_count)
    for i, res_iter in zip(range(ref_imV.shape[0]), res):
        coords[i] = [res_iter[0][0], res_iter[0][1]]
        max_vals[i] = res_iter[1]
    print("done")
else:
    print(
        f"Loading data from coords.txt and /max_vals.txt. This will fail if these do not exist."
    )
    coords = np.loadtxt("coords.txt")
    max_vals = np.loadtxt("max_vals.txt")

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
            warp_imV.shape[0]))#, ceil(warp_imV.shape[0]/cpu_count)
    for i, res_iter in zip(range(warp_imV.shape[0]), res):
        coords_warp[i] = [res_iter[0][0], res_iter[0][1]]
        max_vals_warp[i] = res_iter[1]
    print("done")
else:
    print(
        f"Loading data from coords_warp.txt and /max_vals_warp.txt. This will fail if these do not exist."
    )
    coords_warp = np.loadtxt("coords_warp.txt")
    max_vals_warp = np.loadtxt("max_vals_warp.txt")

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
    plt.savefig('distance_histogram.png')
inds[dist < dist_min] = False
# inds[dist>1000] = False

# save values from finding disks as text files
if(save_all_precompute):
    # os.makedirs(stem_results, exist_ok=True)
    np.savetxt("coords.txt", coords)
    np.savetxt("max_vals.txt", max_vals)
    np.savetxt("coords_warp.txt", coords_warp)####################################################################################
    np.savetxt("max_vals_warp.txt", max_vals_warp)
    np.savetxt("inds.txt", inds)
    np.savetxt("dist.txt", dist)

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
unwarped_test = im_tools.unwarp_im(warp_avg, ab, plot_flag=plot_flag)[0]
plt.imsave("warped_test.png", warp_avg)
plt.imsave("unwarped_test.png", unwarped_test)

# maybe still need to fix gain? using original image?  i don't think so...

np.save('ab_distortion_matrix.npy',ab)
