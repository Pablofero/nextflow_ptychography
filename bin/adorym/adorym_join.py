from pathlib import Path
from tiler import Tiler, Merger
import numpy as np
from jsonargparse import ActionConfigFile, ArgumentParser
from jsonargparse.typing import path_type
import dxchange

Path_nocheck = path_type('rw', docstring='str pointing to a folder', skip_check=True)# for directories
Path_f_nocheck =  path_type('frw', docstring='str pointing to a file', skip_check=True) # for files
# if you want to check use Path_dw or Path_fr from jsonargparse.typing or create custom https://jsonargparse.readthedocs.io/en/stable/index.html#parsing-paths


parser = ArgumentParser(parse_as_dict=True)
parser.add_argument('--cfg', action=ActionConfigFile)
parser.add_argument("--total_tiles_shape", type=str)
parser.add_argument("--recon", type=str ,nargs='+')
params = parser.parse_args()

total_tiles_shape = np.load(params["total_tiles_shape"])
print("total_tiles_shape: ", total_tiles_shape)
print("recon: ", params["recon"])
tile_temp = dxchange.read_tiff(params["recon"][0]+"/delta_ds_1.tiff") # Hack, real value should be passes
print("FIX ME! temp = ...")
print("tile_temp.shape",tile_temp.shape)
probe_temp = dxchange.read_tiff(params["recon"][0]+"/probe_mag_ds_1.tiff")
print("one probe.shape",probe_temp.shape)
probe_mag = np.empty((len(params["recon"]),*probe_temp.shape),dtype=probe_temp.dtype)
probe_phase = np.empty((len(params["recon"]),*probe_temp.shape),dtype=probe_temp.dtype)
tiler = Tiler(data_shape=(*total_tiles_shape,tile_temp.shape[-1]),
              tile_shape=tile_temp.shape)
merger_beta = Merger(tiler)
merger_delta = Merger(tiler)

for t in params["recon"]:
    tile_id = int(t[-1])
    merger_beta.add(tile_id, dxchange.read_tiff(t+"/beta_ds_1.tiff"))
    merger_delta.add(tile_id, dxchange.read_tiff(t+"/delta_ds_1.tiff"))
    probe_mag[tile_id] = dxchange.read_tiff(t+"/probe_mag_ds_1.tiff")[0]
    probe_phase[tile_id] = dxchange.read_tiff(t+"/probe_phase_ds_1.tiff")[0]

final_beta = merger_beta.merge(unpad=True)
final_delta = merger_delta.merge(unpad=True)
dxchange.write_tiff(final_beta,"beta_ds_joined.tiff")
dxchange.write_tiff(final_delta,"delta_ds_joined.tiff")
dxchange.write_tiff(probe_mag,"probe_mag_ds_joined.tiff")
dxchange.write_tiff(probe_phase,"probe_phase_ds_joined.tiff")