from pathlib import Path
import sys
import os.path
from typing import Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import scipy.ndimage as spim
from jsonargparse import ActionConfigFile, ArgumentParser
from jsonargparse.typing import path_type

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from mytools import im_tools, ides_tools

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
parser.add_argument("--px_size_ang_m", type=float, help="the pixel size of the reconstruction in meters",default=1e-10)
parser.add_argument("--rot_ang_deg", type=float, help="rotation angle in degrees", default="0")
parser.add_argument("--extra_vacuum_space", type=int, help="How much vacuum should be put. Measured from the positions bounding box and in integer units of scan_Step_Size",default=0)
###
parser.add_argument("--split_in_subarray", type=int, nargs=2,help="how many tiles in the n rows x m colums, (n, m).", default=[1,1])
parser.add_argument("--overlap",type=int, help="Specifies overlap between subarrays, overlap is in positions.", default=0)
parser.add_argument("--probe_size", type=int, nargs=2,help="Size of the electron probe", default=[-1,-1])
parser.add_argument("--voltage", type=float,help="Microscope Voltage in keV", default=60)
parser.add_argument("--alpha_max", type=float,help="Convergence angle in mrad", default=25)

params = parser.parse_args()

# 4d-dataset
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

Scan_pos = params['scan_pos_list'][0]*params['scan_pos_list'][1]
Scan_Pos_str = str(params['scan_pos_list']).replace('[','').replace(']','').replace(', ','*')
# Scan_pos = eval(params['Scan_Pos_str']);Scan_pos_str = params['Scan_Pos_str'].replace('/','⁄')
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

split_in_subarray_num = params['split_in_subarray']
if split_in_subarray_num == [0,0]:
    split_in_subarray_num =  [params['scan_pos_list'][0],params['scan_pos_list'][1]]
overlap = params['overlap']

def get_width_height_positions(arr):
    ''''
    arr: array with arr[0] = [x,y]
    returns: [width, height]
    '''
    xmin, xmax = np.min(arr[:,0]), np.max(arr[:,0])
    ymin, ymax = np.min(arr[:,1]), np.max(arr[:,1])
    # plt.plot([xmin, xmax],[ymin, ymax])
    return np.array([xmax-xmin,ymax-ymin])

def split_flatten_save_h5(arr, nm, overlap,original_2d_shape, name,debugflag=False):
    '''
    Splits first two dimensions of array arr into n sub arrays, flaten the first dimension, and saves them as name_#
    arr: array should be shape = (a, b,...) with a = 1, but should work for a =/= 1
    nm: tuple, array, how many tiles in the rows x colums aka.  n x m
    overlap: Specifies overlap between subarrays/tiles. Overlap is in positions
    original_2d_shape: 2 tuple/array with the dimensions of the pre-flatten array's first two dimensions
    '''
    print('split_flatten_save_h5\ntiles rows x colums=',nm,'\noverlap in positions = ', overlap)
    nm = np.array(nm)
    arr_ = np.reshape(arr,(*original_2d_shape,*arr.shape[2:])) # de-flaten array so first dimension correspond to realspace
    slices = []
    original_2d_shape = np.array(original_2d_shape)
    nm = np.array(nm)
    dx, dy = original_2d_shape//nm
    restx,resty = original_2d_shape%nm
    addrestx, addresty = 0,0
    for x in range(nm[0]):
        if x == nm[0]-1:
            addrestx=restx
        else:
            addrestx=0
        for y in range(nm[1]):
            if y == nm[1]-1:
                addresty=resty
            else:
                addresty=0
            #bound checking due to overlap
            lowerx= max(x*dx-overlap,0)
            maxx  = min((x+1)*dx+addrestx+overlap,original_2d_shape[0])
            lowery= max(y*dy-overlap,0)
            maxy  = min((y+1)*dy+addresty+overlap,original_2d_shape[1])
            slices.append(np.s_[lowerx:maxx,lowery:maxy])
    debug = []
    debug_shapes = []
    for slice_id, slice_ in zip(range(len(slices)),slices):
        tile = arr_[slice_]
        if debugflag: 
            debug_shapes.append(tile.shape)

        data_ = np.reshape(tile,(1,tile.shape[0]*tile.shape[1],*tile.shape[2:]))
        name_ = name+'_'+str(slice_id)+'.h5'
        debug.append(data_)
        print(slice_)
        with h5py.File(name_,'w') as data_file:
            data_file.create_dataset('exchange/data', data=data_)
    if debugflag: 
        return debug,debug_shapes
    
def split_flatten_save_probe_pos(arr, nm, overlap,original_2d_shape, name,probe_offset=None,scanStepSize=0,param_extra_vacuum_space=0,debugflag=False):
    '''
    Splits first two dimensions of array arr into n sub arrays, flaten the first dimension, and saves them as name_#
    arr: array should be shape = (a, b,...) with a = 1, but should work for a =/= 1
    nm: tuple, array, cut in tiles with shape n x m
    overlap: Specifies overlap between subarrays/tiles. Overlap is in positions
    original_2d_shape: 2 tuple/array with the dimensions of the pre-flatten array's first two dimensions
    probe_offset: adoryms probe is not centered and has (0,0) in the upper left corner, thus a offset is needed to shift the position (probe_offset=probe_size/2)
    scanStepSize: np.array, shape:(2), to convert to pixels
    '''
    extra_vacuum_space = np.int32(param_extra_vacuum_space*scanStepSize) 
    np.save('extra_vacuum_space.npy',extra_vacuum_space)
    np.save('total_tiles_shape.npy',2*extra_vacuum_space+np.int32(np.ceil(get_width_height_positions(arr))))
    print("total_tiles_shape.npy:", 2*extra_vacuum_space+np.int32(np.ceil(get_width_height_positions(arr))))
    print('extra_vacuum_space=',extra_vacuum_space)
    
    print('split_flatten_save_probe\ntiles shape max =',nm,",\noverlap: ",overlap)
    arr_ = np.reshape(arr,(*original_2d_shape,*arr.shape[1:]))
    slices = []
    slices_no_overlap_sub_shape = []
    original_2d_shape = np.array(original_2d_shape)
    nm = np.array(nm)
    dx, dy = original_2d_shape//nm
    restx,resty = original_2d_shape%nm
    addrestx, addresty = 0,0
    for x in range(nm[0]):
        if x == nm[0]-1:
            addrestx=restx
        else:
            addrestx=0
        for y in range(nm[1]):
            if y == nm[1]-1:
                addresty=resty
            else:
                addresty=0
            # bound checking due to overlap
            lowerx= max(x*dx-overlap,0)
            maxx  = min((x+1)*dx+addrestx+overlap,original_2d_shape[0])
            lowery= max(y*dy-overlap,0)
            maxy  = min((y+1)*dy+addresty+overlap,original_2d_shape[1])
            slices.append(np.s_[lowerx:maxx,lowery:maxy])
            print(np.array([lowerx,maxx,lowery,maxy]))

            # calculate local offsets for joining a later stage
            lowerx_= max(x*dx,0)-lowerx
            maxx_  = min((x+1)*dx+addrestx,original_2d_shape[0])-lowerx
            lowery_= max(y*dy,0)-lowery
            maxy_  = min((y+1)*dy+addresty,original_2d_shape[1])-lowery
            slices_no_overlap_sub_shape.append(np.array([lowerx_,maxx_,lowery_,maxy_]))
            print(np.array([lowerx_,maxx_,lowery_,maxy_]))
            print('\n')
    debug = []
    debug_shapes = []

  

    for slice_id, slice_, slice_no_overlap_ in zip(range(len(slices)),slices,slices_no_overlap_sub_shape):
        tile = arr_[slice_]
        np.save('tile_'+str(slice_id)+'_pos_shape.npy', tile.shape, allow_pickle=False)
        data_ = np.reshape(tile,(1,tile.shape[0]*tile.shape[1],*tile.shape[2:]))
        name_ = name+'_'+str(slice_id)+'.npy'
        data_s = np.squeeze(data_)
        position_top_left = data_s.min(axis=0)
        # small_corrections = np.array([-1,-1])*param_extra_vacuum_space*scanStepSize #
        offset = position_top_left+probe_offset-extra_vacuum_space #-np.mod(position_top_left,1)+position_top_left+probe_offset+small_corrections   # modulo makes sure alignment is kept. keep same coordinates after tile cut
        np.save(name_, data_s-offset)
        np.save('tile_'+str(slice_id)+'_offset.npy', offset,allow_pickle=False)
        np.save('tile_'+str(slice_id)+'_slice_no_overlap.npy', slice_no_overlap_,allow_pickle=False)

        shape_bounding_box = get_width_height_positions(data_s)#.shape[0:2]
        print('shape_bounding_box= ',shape_bounding_box)
        shape_bounding_box+=2*extra_vacuum_space #TODO extra padding, implement parameter
        print('shape_bounding_box with extra= ',shape_bounding_box,'|to int= ',np.int32(np.ceil(shape_bounding_box)))
        np.save('tile_'+str(slice_id)+'_shape_pixels.npy', np.int32(np.ceil(shape_bounding_box)),allow_pickle=False)
    if debugflag: 
        return debug,debug_shapes
####################################### calculate probe positions ########################################### 
N_scan_x=params['scan_pos_list'][0]
N_scan_y=params['scan_pos_list'][1]
scan_step_size_x_m=params['scan_Step_Size_x_A'] * 1e-10 
scan_step_size_y_m=params['scan_Step_Size_x_A'] * 1e-10
px_size_m = params['px_size_ang_m']
np.save('pixel_size.npy', px_size_m)
rot_ang = np.pi/180*params['rot_ang_deg']
param_extra_vacuum_space  = params['extra_vacuum_space']
scanStepSize_arr = np.array([scan_step_size_x_m,scan_step_size_y_m])/px_size_m
print('scanStepSize_arr: ',scanStepSize_arr)

out_name = 'beam_pos_'+str(params['rot_ang_deg'])+'deg_'
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
plt.scatter(xy[0,:], xy[1,:],label='original')
plt.scatter(xy_rot[0,:], xy_rot[1,:],label='rotated')
plt.xlabel('m')
plt.ylabel('m')
plt.title('no probe offset')
plt.legend()
plt.savefig('beam_pos_no_probe_offset.png')

xy_rot /= px_size_m

xy_rot = xy_rot -  np.min(xy_rot, axis=1)[:,None]

plt.figure(11, clear=True)
plt.grid()
plt.xlabel('px')
plt.ylabel('px')
plt.scatter(xy_rot[0,:], xy_rot[1,:], c=np.linspace(0,1,N_scan_x*N_scan_y))
plt.savefig('beam_pos2.png')
plt.clf()
# print(xy_rot)
#saving done at the end of document
####################################### calculate probe positions END ########################################### 

values_shape_2d = values.shape[0:2]
dimension_r = values.shape[2]
dimension_c = values.shape[3]

print('crop_inds[0]:(values.shape[2]-crop_inds[1]), crop_inds[2]:(values.shape[3]-crop_inds[3])',crop_inds[0],':',(values.shape[2]-crop_inds[1]),'    ', crop_inds[2],':',(values.shape[3]-crop_inds[3]))
values = np.transpose(values, (0,1,2,3)) # currently does nothing
values =  np.reshape(values,(1,values.shape[0]*values.shape[1],dimension_r,dimension_c))
# values *= 214183.488


# calculate com shit corresponding to average diffraction pattern, then apply to every diffraction pattern in parallel
_, shifts = im_tools.shift_im_com(im_tools.bin_image(np.mean(values, axis=(0,1)), binFactor), thresh=1)

np.savetxt("com_shift.txt",shifts)

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
x_cm,y_cm = im_tools.com_im(np.mean(values_new, axis=(0,1)), thresh=1, plot_flag=True)
plt.savefig('final_mean_COM_{}_{}.png'.format(x_cm,y_cm))


# Write data to HDF5
# fname = path.parent / (path.stem+out_name_append+'.h5')
split_flatten_save_h5(values_new, split_in_subarray_num,overlap ,values_shape_2d, out_name_append)# arr, nm,original_2d_shape, name,safe_HDF5=False,scanStepSize=None
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

split_flatten_save_probe_pos(xy_rot.T, split_in_subarray_num,overlap , params['scan_pos_list'], out_name, probe_offset = probe_size/2, scanStepSize=scanStepSize_arr,  param_extra_vacuum_space = param_extra_vacuum_space)# arr, nm,original_2d_shape, name,safe_HDF5=False,scanStepSize=None


# np.sqrt(dat)
# wavefront = np.mean(np.abs(dat), axis=(0, 1))


## generate probe and apply phase shift coreponding to com shift
N = np.max(probe_size)
probe_generated = ides_tools.generate_probe(dx=px_size_m*10**10, N=N, voltage=params["voltage"], alpha_max=params["alpha_max"], df=0,C3=0,C5=0,C7=0)#TODO pass more parameters # dx is in angstroms
plt.imsave("probe_generated_NO_SHIFT_angle.png",np.angle(probe_generated),cmap="gray")
plt.imsave("probe_generated_NO_SHIFT_abs.png",np.abs(probe_generated),cmap="gray")

kx = np.linspace(-np.floor(N/2), np.ceil(N/2)-1,N)
# kx = np.fft.fftfreq(N, 1/N)
[kX,kY] = np.meshgrid(kx,kx)
shifftx, shiffty = shifts
# add ramp in the fase to compensate for shift of COM
probe_generated = probe_generated*np.exp(-1j/N*(shifftx*kX+shiffty*kY))# 1/N is coused by np fft normalization
plt.imsave("probe_generated_real.png",np.real(probe_generated),cmap="gray")
plt.imsave("probe_generated_imag.png",np.imag(probe_generated),cmap="gray")
plt.imsave("probe_generated_abs.png",np.abs(probe_generated),cmap="gray")
plt.imsave("probe_generated_angle.png",np.angle(probe_generated),cmap="gray")
np.save("probe_generated.npy",probe_generated)

print('Done!')