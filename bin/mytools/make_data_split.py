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
parser.add_argument("--bin_Factor", type=int, help="diffraction pattern binning", default=4)
parser.add_argument("--beamstop_thresh", type=float, help="threshold applied over diffraction pattern mean to calculate a beamstop", default=0.01)
parser.add_argument("--crop_pad_diffraction", type=int, nargs=4,help="crop or pad the diffraction pattern, negative pads with edge values, positives crops. [crop_pad_row_first, crop_pad_row_last, crop_pad_colum_first, crop_pad_colum_last]", default=[0, 0, 0, 0])
parser.add_argument("--scan_pos_list", type=int, nargs=2,help="list defining how many scan position there are by giving x and y amounts", default=[75,75])
parser.add_argument("--Path_2_Unwarped", type=Path_f_nocheck)
parser.add_argument("--rotate_180", type=bool, help="Rotate patterns 180 degrees", default=False)
parser.add_argument("--transpose_data", type=bool, help="transpose potential realspace positions", default=False)
parser.add_argument("--out_name_append", type=str, help='start with "_"!, parameters will be the autoappended',default='_adorym_original')#og: _adorym_original_reduced75x75_shifted_bin4_rotated180 # TODO
parser.add_argument("--cpu_count", type=int, help="amount of cores to be used", required=True)
###
parser.add_argument("--scan_Step_Size_x_A", type=float, help="")
parser.add_argument("--scan_Step_Size_y_A", type=float, help="")
parser.add_argument("--rot_ang_deg", type=float, help="rotation angle in degrees", default="0")
parser.add_argument("--px_size_ang_m", type=float, help="the pixel size of the reconstruction in meters",default=1e-10)
###
parser.add_argument("--split_in_subarray_num_list", type=int, nargs=2,help="list defining how many subarrays to split the data into", default=[1,1])
parser.add_argument("--probe_size", type=int, nargs=2,help="Size of the electron probe", default=[-1,-1])
params = parser.parse_args()

#4d-dataset
path = Path(str(params['Path_2_Unwarped']));path.is_file()
values = np.load(path) # has shape [probe_pos_x, probe_pos_y, dimension_r, dimension_c]
if params['transpose_data']:
    values = values.transpose(1,0,2,3)
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
    values = np.pad(values, ((0,0),(0,0),-crop_inds[:2], -crop_inds[2:]), mode='constant')
elif np.all(crop_inds>=0):
    # then we crop
    values = values[:, :, crop_inds[0]:(values.shape[2]-crop_inds[1]), crop_inds[2]:(values.shape[3]-crop_inds[3])]
else:
    raise ValueError('Padding and cropping at the same time is not allowed... yet. Please change crop_ parameters such that they are all positive or negative, not mixed.')
cpu_count = params["cpu_count"]
binFactor = params['bin_Factor']
split_in_subarray_num = params['split_in_subarray_num_list']

Scan_pos = params['scan_pos_list'][0]*params['scan_pos_list'][1]
Scan_Pos_str = str(params['scan_pos_list']).replace('[','').replace(']','').replace(', ','*')
# Scan_pos = eval(params['Scan_Pos_str']);Scan_pos_str = params['Scan_Pos_str'].replace('/','⁄')
np.save('total_tiles_shape.npy',np.array(params['scan_pos_list']),)
size=values.shape[2]*values.shape[3]*Scan_pos
out_name_append = params['out_name_append']
if out_name_append[0] != '_':
    raise ValueError('out_name_append must start with "_" but out_name_append is:"'+str(out_name_append)+'"')
out_name_append+=('_rotated180patterns'*params['rotate_180'])
out_name_append+=('_transposed'*params['transpose_data'])
out_name_append+='_Dim'+str(values.shape[2])+str(values.shape[3])
out_name_append+='_Bin'+str(binFactor)
out_name_append+='_scan_pos'+Scan_Pos_str
out_name_append+='_size'+str(size)
print('out_name_append=',out_name_append)




def split_flatten_save(arr, nm,original_2d_shape, name,safe_HDF5=False,scanStepSize=None,probe_offset=None):
    '''
    Splits first two dimensions of array arr into nm[0]*nm[1] sub arrays, flaten the first dimension, and saves them as name_#
    arr: array should be shape = (a, b,...) with a = 1, but should work for a =/= 1
    nm: tuple, array, cut in n x m tiles
    original_2d_shape: 2 tuple/array with the dimensions of the pre-flatten array's first two dimensions
    scanStepSize: np.array, shape:(2), in pixels
    probe_offset: adoryms probe is not centered and has (0,0) in the upper left corner, thus a offset is needed to shift the position
    '''
    # plt.clf()
    # fig,ax = plt.subplots(n, m,squeeze=False)
    # max_val = arr.max()
    n,m = nm
    print('n m =',n,m)
    if len(arr.shape)>2:
        arr_ = np.reshape(arr,(*original_2d_shape,*arr.shape[2:]))
    else:
        arr_ = np.reshape(arr,(*original_2d_shape,*arr.shape[1:]))
    s  = np.array_split(arr_,n) # returns list
    counter = 0
    for i in range(len(s)):
        s2 = np.array_split(s[i],m,axis=1) # returns list
        for j in range(len(s2)):
            data_shape_2d = s2[j].shape[0:2]
            data_ = np.reshape(s2[j],(1,s2[j].shape[0]*s2[j].shape[1],*s2[j].shape[2:]))
            if safe_HDF5:
                name_ = name+'_'+str(counter)+'.h5'
                with h5py.File(name_,'w') as data_file:
                    data_file.create_dataset('exchange/data', data=data_)
                np.save('tile_'+str(counter)+'_shape.npy', np.int32(np.ceil((data_shape_2d*scanStepSize))),allow_pickle=False)
                print('tile_'+str(counter)+'_shape.npy without stepsize:', data_shape_2d,'tile_'+str(counter)+'_shape.npy with stepsize:',np.int32(np.ceil(data_shape_2d*scanStepSize)), '\ts2[j].shape:',s2[j].shape)
            else:
                name_ = name+'_'+str(counter)+'.npy'
                data_s = np.squeeze(data_)
                position_top_left = data_s.min(axis=0)
                np.save(name_, data_s-(position_top_left+probe_offset))
                np.save('tile_'+str(counter)+'_offset.npy', (position_top_left+probe_offset),allow_pickle=False)
            counter += 1
            #s2[j] is now one of the subarrays, the other dimensions are untouched (aka. third, fourth..)
            # ax[i][j].imshow(s2[j].mean(axis=(-1,-2), vmin=0., vmax=max_val,aspect="auto")
            # ax[i][j].set_xlabel(str(i)+' '+str(j))
            





####################################### calculate probe positions ########################################### 
N_scan_x=params['scan_pos_list'][0]
N_scan_y=params['scan_pos_list'][1]
scan_step_size_x_m=params['scan_Step_Size_x_A'] * 1e-10 
scan_step_size_y_m=params['scan_Step_Size_x_A'] * 1e-10
rot_ang = np.pi/180*params['rot_ang_deg']
px_size_m = params['px_size_ang_m']
scanStepSize_arr = np.array([scan_step_size_x_m,scan_step_size_y_m])/px_size_m
print('scanStepSize_arr: ',scanStepSize_arr)

out_name = 'beam_pos_'+str(rot_ang)+'deg_'
out_name += 'Scan'+str(N_scan_x)+'x'+str(N_scan_y)+'_'
out_name += 'StepsSize'+str(scan_step_size_x_m)+'x'+str(scan_step_size_y_m)

ppx = np.arange(N_scan_x)
ppy = np.arange(N_scan_y)

ppY, ppX = np.meshgrid(ppx, ppy, indexing='ij')

R = np.asarray(
    [[np.cos(rot_ang), -np.sin(rot_ang)],
     [np.sin(rot_ang), np.cos(rot_ang)]]
)

xy = np.vstack((ppX.ravel() * scan_step_size_x_m, ppY.ravel() * scan_step_size_y_m))

print(xy)

xy_rot = R @ xy

plt.figure(10,clear=True)
plt.scatter(xy[0,:], xy[1,:])
plt.scatter(xy_rot[0,:], xy_rot[1,:])
plt.savefig('beam_pos.png')

xy_rot /= px_size_m

xy_rot = xy_rot -  np.min(xy_rot, axis=1)[:,None]

plt.figure(11, clear=True)
plt.scatter(xy_rot[0,:], xy_rot[1,:], c=np.linspace(0,1,N_scan_x*N_scan_y))
plt.savefig('beam_pos2.png')
plt.clf()
# print(xy_rot)
#savin done at the end of document
####################################### calculate probe positions END ########################################### 

values_shape_2d = values.shape[0:2]
dimension_r = values.shape[2]
dimension_c = values.shape[3]

print('crop_inds[0]:(values.shape[2]-crop_inds[1]), crop_inds[2]:(values.shape[3]-crop_inds[3])',crop_inds[0],':',(values.shape[2]-crop_inds[1]),'    ', crop_inds[2],':',(values.shape[3]-crop_inds[3]))
values = np.transpose(values, (0,1,2,3)) # currently does nothing
values =  np.reshape(values,(1,values.shape[0]*values.shape[1],dimension_r,dimension_c))
# values *= 214183.488


_, shifts = im_tools.shift_im_com(im_tools.bin_image(np.mean(values, axis=(0,1)), binFactor), thresh=1)

values_new = np.zeros((1, Scan_pos, int(dimension_r/binFactor), int(dimension_c/binFactor))) #TODO don't need to "reserve" the memory, right now used only as template (assumes OS is smart and doesn't actually reserve that memory)

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
# fname = path.parent / (path.stem+out_name_append+'.h5')
split_flatten_save(values_new, split_in_subarray_num ,values_shape_2d, out_name_append, safe_HDF5=True,scanStepSize=scanStepSize_arr)# arr, nm,original_2d_shape, name,safe_HDF5=False,scanStepSize=None
# with h5py.File(fname,'w') as data_file:
#     data_file.create_dataset('exchange/data', data=values_new)
#     data_file.create_dataset('metadata/probe_pos_px', data=probe_pos_px)

# save beamstop
fnameb = path.parent/(path.stem+out_name_append+'_beamstop.npy')#TODO
np.save(fnameb, np.mean(values_new, axis=(0,1))>params['beamstop_thresh'])

# save some validation images
plt.imsave(path.parent/(path.stem+out_name_append+'_beamstop.png'), np.mean(values_new, axis=(0,1))>.01)
plt.imsave(path.parent/(path.stem+out_name_append+'_final_mean_im.png'), np.mean(values_new, axis=(0,1))**.25)

#probe
if params['probe_size'][0] > 0: #custom probe
    probe_size = np.array(params['probe_size'])
else:
    probe_size = np.array(values_new.shape[-2:])
np.save('probe_size.npy', probe_size)

split_flatten_save(xy_rot.T, split_in_subarray_num , params['scan_pos_list'], out_name, probe_offset = probe_size/2)# arr, nm,original_2d_shape, name,safe_HDF5=False,scanStepSize=None