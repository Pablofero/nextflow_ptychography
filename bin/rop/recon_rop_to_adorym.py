import numpy as np
from pathlib import Path
from jsonargparse import ActionConfigFile, ArgumentParser
import struct
import dxchange

def bin_to_tiff(path,name,dimension,index,f=None):
    file = open(path, 'rb')
    data = file.read()
    values = struct.unpack(dimension * dimension * 'f', data)
    if f is not None:
        values = f(values)
    values =  np.reshape(values,(dimension,dimension))
    dxchange.write_tiff(values,str(path.parent/("recon_"+str(index))/(name+".tiff")), dtype='float32', overwrite=True)
    file.close()

parser = ArgumentParser(parse_as_dict=True)
parser.add_argument('--cfg', action=ActionConfigFile)
parser.add_argument("--tile_shape", type=str )
parser.add_argument("--PotentialImag", type=str )
parser.add_argument("--PotentialReal", type=str )
parser.add_argument("--probe_size", type=str )
parser.add_argument("--ProbeImag", type=str )
parser.add_argument("--ProbeReal", type=str )
parser.add_argument("--positions", type=str )
params = parser.parse_args()

recon_dimension=np.load(params["tile_shape"]).max()
probe_dimension=np.load(params["probe_size"]).max()
index = int(Path(params["positions"]).stem.split("_")[-1])

bin_to_tiff(Path(params["PotentialImag"]),"PotentialImag",recon_dimension,index)
bin_to_tiff(Path(params["PotentialReal"]),"PotentialReal",recon_dimension,index)
bin_to_tiff(Path(params["PotentialReal"]),"delta_ds_1",recon_dimension,index,lambda x:x)

bin_to_tiff(Path(params["ProbeImag"]),"ProbeImag",probe_dimension,index)
bin_to_tiff(Path(params["ProbeReal"]),"ProbeReal",probe_dimension,index)
