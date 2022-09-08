from pathlib import Path
import sys
import os.path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import scipy.ndimage as spim
from jsonargparse import ActionConfigFile, ArgumentParser
from jsonargparse.typing import path_type

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from mytools import im_tools

Path_nocheck = path_type('rw', docstring='str pointing to a folder', skip_check=True)# for directories
Path_f_nocheck =  path_type('frw', docstring='str pointing to a file', skip_check=True) # for files
# if you want to check use Path_dw or Path_fr from jsonargparse.typing or create custom https://jsonargparse.readthedocs.io/en/stable/index.html#parsing-paths


parser = ArgumentParser(parse_as_dict=True)
parser.add_argument('--cfg', action=ActionConfigFile)
parser.add_argument("--bin_Factor", type=int, help="", default=4)
parser.add_argument("--crop_pad_diffraction", type=int, nargs=4,help="crop or pad the diffraction pattern, negative pads with edge values, positives crops. [crop_pad_row_first, crop_pad_row_last, crop_pad_colum_first, crop_pad_colum_last]", default=[0, 0, 0, 0])
parser.add_argument("--scan_pos_list", type=int, nargs=2,help="list defining how many scan position there are by giving x and y amounts", default=[75,75])
parser.add_argument("--Path_2_Unwarped", type=Path_f_nocheck)
parser.add_argument("--rotate_180", type=bool, help="Rotate patterns 180 degrees", default=False)
parser.add_argument("--out_name_append", type=str, help='start with "_"!, parameters will be the autoappended',default='_adorym_original')#og: _adorym_original_reduced75x75_shifted_bin4_rotated180 # TODO
parser.add_argument("--cpu_count", type=int, help="amount of cores to be used", required=True)
params = parser.parse_args()

path = Path(str(params['Path_2_Unwarped']));path.is_file()
values = np.load(path) # has shape [probe_pos_x, probe_pos_y, dimension_r, dimension_c]
# values = np.load(path, 'r') # has shape [probe_pos_x, probe_pos_y, dimension_r, dimension_c]
#values = values[125:200,50:125,:,:] # TODO implement crop in realspace 
# crop_inds = [params['crop_r_before'], params['crop_r_after'], params['crop_c_before'], params['crop_c_after']]
print('values.shape',values.shape)
crop_inds = np.asarray(params['crop_pad_diffraction'])

if np.all(crop_inds==0):
    # do nothing
    pass
elif np.all(crop_inds<=0):
    # then we pad
    values = np.pad(values, ((0,0),(0,0),-crop_inds[:2], -crop_inds[2:]), mode='edge')
elif np.all(crop_inds>=0):
    # then we crop
    values = values[:, :, crop_inds[0]:(values.shape[2]-crop_inds[1]), crop_inds[2]:(values.shape[3]-crop_inds[3])]
else:
    raise ValueError('Padding and cropping at the same time is not allowed... yet. Please change crop_ parameters such that they are all positive or negative, not mixed.')
cpu_count = params["cpu_count"]
binFactor = params['bin_Factor']
Scan_pos = params['scan_pos_list'][0]*params['scan_pos_list'][1]
Scan_Pos_str = str(params['scan_pos_list']).replace('[','').replace(']','').replace(', ','*')
# Scan_pos = eval(params['Scan_Pos_str']);Scan_pos_str = params['Scan_Pos_str'].replace('/','⁄')
size=values.shape[2]*values.shape[3]*Scan_pos
out_name_append = params['out_name_append']
if out_name_append[0] != '_':
    raise ValueError('out_name_append must start with "_" but out_name_append is:"'+str(out_name_append)+'"')
out_name_append+=('_rotated180'*params['rotate_180'])
out_name_append+='_Dim'+str(values.shape[2])+str(values.shape[3])
out_name_append+='_Bin'+str(binFactor)
out_name_append+='_scan_pos'+Scan_Pos_str
out_name_append+='_size'+str(size)
print('out_name_append=',out_name_append)

dimension_r = values.shape[2]
dimension_c = values.shape[3]

print('crop_inds[0]:(values.shape[2]-crop_inds[1]), crop_inds[2]:(values.shape[3]-crop_inds[3])',crop_inds[0],':',(values.shape[2]-crop_inds[1]),'    ', crop_inds[2],':',(values.shape[3]-crop_inds[3]))
values = np.transpose(values, (0,1,2,3)) # currently does nothing
values =  np.reshape(values,(1,Scan_pos,dimension_r,dimension_c))
# values *= 214183.488


_, shifts = im_tools.shift_im_com(im_tools.bin_image(np.mean(values, axis=(0,1)), binFactor), thresh=1)

values_new = np.zeros((1, Scan_pos, int(dimension_r/binFactor), int(dimension_c/binFactor))) #TODO don't need to reserve the memory, right now used only as template

print('shifting data')
print('creating RawArray...')
values_new_ctypes_arr = mp.RawArray(np.ctypeslib.as_ctypes_type(values_new.dtype),values_new.size)# Note that setting and getting an element is potentially non-atomic, ok in this case though as each process reads/writes data independently to the others!
print('values_new.dtype: ',values_new.dtype)
print('np.ctypeslib.as_ctypes_type(values_new.dtype): ',np.ctypeslib.as_ctypes_type(values_new.dtype))
print('values_new.size: ',values_new.size)
print('values_new.shape: ',values_new.shape)
values_new_ctypes_nparray = np.ctypeslib.as_array(values_new_ctypes_arr,values_new.shape)
values_new_ctypes_nparray.shape = values_new.shape
print('values_new_ctypes_nparray.shape: ', values_new_ctypes_nparray.shape)
print('values.shape: ', values.shape)


if params['rotate_180']: #rotate by 180 degrees?
    slicee = (slice(None,None,-1),slice(None,None,-1)) #[::-1,::-1]
else:
    slicee = None #[:,:]

def shift_bin(i):
    global values_ctypes_nparray
    values_new_ctypes_nparray[0,i,:,:] = spim.shift(im_tools.bin_image(values[0,i,:,:], binFactor), shifts, order=1)[slicee]
print('cpu count:',cpu_count)
with mp.Pool(cpu_count) as p:
    p.map(shift_bin, range(values_new.shape[1]))#, ceil(dataV.shape[0]/cpu_count)

values_new = values_new_ctypes_nparray

# for i in range(values.shape[1]):#tqdm was here
#     if params['rotate_180']:
#         values_new[0,i,:,:] = spim.shift(im_tools.bin_image(values[0,i,:,:], binFactor), shifts, order=1)[::-1,::-1]
#     else:
#         values_new[0,i,:,:] = spim.shift(im_tools.bin_image(values[0,i,:,:], binFactor), shifts, order=1)

print('final mean COM')
im_tools.com_im(np.mean(values_new, axis=(0,1)), thresh=1, plot_flag=True)



# Write data to HDF5
fname = path.parent / (path.stem+out_name_append+'.h5')
with h5py.File(fname,'w') as data_file:
    data_file.create_dataset('exchange/data', data=values_new)

# save beamstop
fnameb = path.parent/(path.stem+out_name_append+'_beamstop.npy')#TODO
np.save(fnameb, np.mean(values_new, axis=(0,1))>.01)

# save some validation images
plt.imsave(path.parent/(path.stem+out_name_append+'_beamstop.png'), np.mean(values_new, axis=(0,1))>.01)
plt.imsave(path.parent/(path.stem+out_name_append+'_final_mean_im.png'), np.mean(values_new, axis=(0,1))**.25)
