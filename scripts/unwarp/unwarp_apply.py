import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from math import ceil
import os

# setup import of own tools.
# append the path to the folder containing im_tools.py and __init__.py
sys.path.append(os.path.split(sys.path[0])[0]) # /scripts
# print("sys.version is", sys.version)
# print('sys.path[0]',f'{sys.path[0]}')
from mytools import im_tools
################################### -mod- ############################################################
################################### UNFINISHED ############################################################

# for config import, commandline and json/yaml file
from jsonargparse import ArgumentParser, ActionConfigFile
# for defining custom file_path type and geting allredy define ones
from jsonargparse.typing import path_type, Path_fr, Path_dw
# for specifying type of input, checks things before anything is actually needed https://jsonargparse.readthedocs.io/en/stable/index.html#type-hints
from typing import DefaultDict, List
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
# more about how to use: https://jsonargparse.readthedocs.io/en/stable/#parsing-paths
parser.add_argument("--file", type=Path_f_nocheck, help="raw data stack path - 4D numpy files extracted from Swift", required=True)
parser.add_argument("--ab", type=Path_f_nocheck, help="ab distortion matrix")
parser.add_argument("--use_json", type=bool, help="this is currently in beta and not to be used without explanation from Tom, in which order the data should be processsed. These require variables defined from find_points_ref and find_points_warp", default=False)
parser.add_argument("--cpu_count", type=int, help="amount of cores to be used", required=True)
#TODO flag for plotting

#######################################################################################################
#######################################################################################################


# plot_flag = False


################################### -mod-  ############################################################
################################### UNFINISHED ############################################################

params = parser.parse_args()
# TODO Nextflow? get usable processors: len(os.sched_getaffinity(0))
cpu_count = params["cpu_count"]
file = Path(params["file"])
ab = Path(params["ab"])
use_json = params["use_json"]
#######################################################################################################

#######################################################################################################
# # fit surface, ab is the distortion matrix
# warp_avg = np.mean(warp_im, axis=(0, 1))
# ab = im_tools.find_ab(
#     coords[inds, 0] - np.mean(coords[inds, 0]),
#     coords[inds, 1] - np.mean(coords[inds, 1]),
#     coords_warp[inds, 0],
#     coords_warp[inds, 1],
# )

# if use_json:
#     ab = im_tools.find_ab(x[inds] - np.mean(x[inds]), y[inds] -
#                           np.mean(y[inds]), coords_warp[inds, 0], coords_warp[inds, 1])

# # this command unwarps the image
# im_tools.unwarp_im(warp_avg, ab, plot_flag=plot_flag)
# if plot_flag:
#     plt.pause(5)  # HACK it was plt.pause(0.01), it disapears in my machine

# maybe still need to fix gain? using original image?  i don't think so

print(ab)
ab = np.load(ab)#load distortion matrix
# now we unwarp the 4D stacks of actual data
data = np.load(file, mmap_mode="r")
print("data.shape:", data.shape)
# print("padding real data...") #HACK no needed?: pad_size = ((0, 0), (0, 0), (0, 0), (0, 0))
# data = np.pad(
#     data, pad_size
# )  # this is perhaps unnecessary now that my padding function can handle 2D images. This step takes a lot of memory
data_mean = np.mean(data, axis=(0, 1))

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
print('cpu count:',cpu_count)
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

plt.imsave(file.parent / "warped_mean.png", data_mean**.25)
plt.imsave(file.parent / "unwarped_mean.png", data_unwarp_mean**.25)
#plt.figure(34, clear=True)
#plt.imshow(data_unwarp_mean ** 0.25)

#data_out.flush()
f_out = file.parent / (file.stem + "_unwarped.npy")
if use_json:
    f_out = file.parent / (file.stem + "_unwarped_json.npy")
np.save(f_out, data_out)
#-# np.save(f_out, data)
print(f"saved to {f_out}")

