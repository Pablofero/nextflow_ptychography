import sys
import numpy as np
import struct
import h5py
from pathlib import Path
from jsonargparse import ActionConfigFile, ArgumentParser

parser = ArgumentParser(parse_as_dict=True)
parser.add_argument('--cfg', action=ActionConfigFile)
parser.add_argument("--data", type=str )
parser.add_argument("--positions", type=str )
parser.add_argument("--probe_size", type=str )
params = parser.parse_args()

data_path = params["data"]
beam_positions_path = Path(params["positions"])
probe_size = np.load(params["probe_size"])# 2D

## data
output_path = Path(data_path).with_suffix(".bin")
#Open HDF5 Diffraction Data
h5dataset = h5py.File(data_path,'r')
data = h5dataset['exchange'].get('data')
data = np.asarray(data)[0]
print("data.shape",data.shape)
if data.shape[1] != data.shape[2]:
    	raise Exception("CBEDDim is not square, see last two dimensions of data.shape is "+str(data.shape)+"")
np.save("CBEDDim.npy",np.max(data.shape[1:3]))

positions_n = data.shape[0]
dimension_CBED = data.shape[1]*data.shape[2]
#Rescale data for ROP
data /= (np.sum(data) / positions_n)
#Rotate every DP and save it
with open(output_path, 'ab') as output_file:
	for dp in range(positions_n):
		data_dp = np.rot90(data[dp,:,:], 2)
		data_dp = np.ravel(data_dp)
		data_dp = struct.pack(dimension_CBED*'f', *data_dp)
		output_file.write(data_dp)

# positions
# np.save(beam_positions_path.stem+"_rop_"+beam_positions_path.stem.split("_")[-1]+".npy",np.load(beam_positions_path)-probe_size/2)

print("Done!")