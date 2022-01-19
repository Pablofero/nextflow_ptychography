from os import name
import h5py
from h5py._hl.base import default_lapl
import numpy as np
#from tqdm import tqdm
import struct
import scipy.ndimage as spim
import sys,os
#sys.path.append("/Users/Tom/Documents/Research/code/pyscripts/")
from mytools import im_tools
from mytools import ides_tools
import matplotlib.pyplot as plt
from pathlib import Path

from jsonargparse import ArgumentParser, ActionConfigFile
from jsonargparse.typing import path_type
Path_nocheck = path_type('rw', docstring='str pointing to a folder', skip_check=True)# for directories
Path_f_nocheck =  path_type('frw', docstring='str pointing to a file', skip_check=True) # for files
# if you want to check use Path_dw or Path_fr from jsonargparse.typing or create custom https://jsonargparse.readthedocs.io/en/stable/index.html#parsing-paths


parser = ArgumentParser(parse_as_dict=True)
parser.add_argument('--cfg', action=ActionConfigFile)
parser.add_argument("--Dimension_r", type=int, help="", default=256)
parser.add_argument("--Dimension_c", type=int, help="", default=256)
parser.add_argument("--binFactor", type=int, help="", default=4)
parser.add_argument("--ScanPos_str", type=str, help="evaluates the string, use with care!", default=75**2)
parser.add_argument("--Path2Unwarped", type=Path_f_nocheck)
parser.add_argument("--out_name_append", type=str, help='start with "_"!, parameters will be the autoappended',default='_adorym_original')#og: _adorym_original_reduced75x75_shifted_bin4_rotated180 # TODO
params = parser.parse_args()


Dimension_r = params['Dimension_r']
Dimension_c = params['Dimension_c']
binFactor = params['binFactor']
ScanPos = eval(params['ScanPos_str']);ScanPos_str = params['ScanPos_str'].replace('/','⁄')
size=Dimension_r*Dimension_c*ScanPos
path = Path(str(params['Path2Unwarped']));path.is_file()
out_name_append = params['out_name_append']
if out_name_append[0] != '_':
    raise ValueError('out_name_append must start with "_" but out_name_append is:"'+str(out_name_append)+'"')
out_name_append+='_Dim'+str(Dimension_r)+str(Dimension_c)
out_name_append+='_Bin'+str(binFactor)
out_name_append+='_ScanPos'+ScanPos_str
out_name_append+='_size'+str(size)
print('out_name_append=',out_name_append)

# slice = 0
# f = open(path, 'rb').read()
# values = struct.unpack(size*'f', f)
values = np.load(path, 'r')
#print(f'{str(path)=},{values.shape=}')
#values = values[125:200,50:125,:,:]
values = np.transpose(values, (0,1,2,3))
values =  np.reshape(values,(1,ScanPos,Dimension_r,Dimension_c))
# values *= 214183.488


_, shifts = im_tools.shift_im_com(im_tools.bin_image(np.mean(values, axis=(0,1)), binFactor), thresh=1)

values_new = np.zeros((1, ScanPos, int(Dimension_r/binFactor), int(Dimension_c/binFactor)))
print('shifting data')
for i in range(values.shape[1]):#tqdm was here
        values_new[0,i,:,:] = spim.shift(im_tools.bin_image(values[0,i,:,:], binFactor), shifts, order=1)[::-1,::-1] #rotate by 180 degrees? # TODO

print('final mean COM')
im_tools.com_im(np.mean(values_new, axis=(0,1)), thresh=1, plot_flag=True)



# Write data to HDF5
fname = path.parent / (path.stem+out_name_append+'.h5')
with h5py.File(fname,'w') as data_file:
    data_file.create_dataset('exchange/data', data=values_new)

# save beamstop
fnameb = path.parent/(path.stem+out_name_append+'_beamstop.npy')#TODO
np.save(fnameb, np.mean(values_new, axis=(0,1))>.01)

plt.figure(83, clear=True)
plt.imshow(np.mean(values_new, axis=(0,1))>.01)
#plt.show(block=True)

# # 333.68
# pos = ides_tools.gen_beam_positions((75,75), 20e-12, 190.11,write_txt=False)
# pos /= 1.009e-11 # pixel size
# pos -= np.min(pos) 
# pos += 1 # small offset
# np.save('/Users/Tom/Documents/Research/Data/2021/2021-02-16_bilayer_graphene_ptychography_second_library/exports/Spectrum Image (Dectris)_100mrad_pelz_unfiltered_21.25pm_unwarped_json_adorym_bin4_pos.npy',pos)

#B = h5py.File('/home/marcel/Desktop/Gitlab/adorym/demos/cameraman_pos_error/data_cameraman_err_10.h5','r')
#B = h5py.File('/home/marcel/Desktop/Gitlab/adorym/demos/cone_256_foam_ptycho/data_cone_256_foam_1nm.h5','r')
# B = h5py.File('/Users/Tom/Documents/Research/Data/2021/2021-02-16_bilayer_graphene_ptychography_second_library/exports/Spectrum Image (Dectris)_100mrad_pelz_unfiltered_21.25pm_unwarped_json_adorym_original_reduced75x75_shifted_bin4.h5','r')

# data = list(B['exchange']['data'])[0]
# print(data.shape)
