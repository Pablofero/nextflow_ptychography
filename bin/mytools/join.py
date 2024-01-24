from pathlib import Path
from jsonargparse import ActionConfigFile, ArgumentParser
from jsonargparse.typing import path_type
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from PIL import Image
from skimage import draw
from scipy import signal
px = 1/plt.rcParams['figure.dpi']  # pixel in inches aka  inches/dot

Path_nocheck = path_type('rw', docstring='str pointing to a folder', skip_check=True)# for directories
Path_f_nocheck =  path_type('frw', docstring='str pointing to a file', skip_check=True) # for files
# if you want to check use Path_dw or Path_fr from jsonargparse.typing or create custom https://jsonargparse.readthedocs.io/en/stable/index.html#parsing-paths


parser = ArgumentParser(parse_as_dict=True)
parser.add_argument('--cfg', action=ActionConfigFile)
parser.add_argument("--recon", type=str ,nargs='+')
parser.add_argument("--positions", type=str ,nargs='+')
parser.add_argument("--tile_shape", type=str ,nargs='+')
parser.add_argument("--tile_offset", type=str ,nargs='+')
parser.add_argument("--slice_no_overlap", type=str ,nargs='+')
parser.add_argument("--tile_no_overlab_sub_shape", type=str ,nargs='+')
parser.add_argument("--total_tiles_shape", type=str)
parser.add_argument("--extra_vacuum_space", type=str)
parser.add_argument("--clear_extend", type=int, default=10, help="How much to delete from outside the tile (to be automated), expressed for now in arbitrary units")
parser.add_argument("--half_window_lenght", type=int, default=1, help="How big should the windowing be, expressed for now in arbitrary units")
parser.add_argument("--window_name", type=str, default="hann", help="What windowing function to smooth the overlap of the tiles. Choose from scipy.signal.windows.get_window (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html#scipy.signal.windows.get_window)")

params = parser.parse_args()

tiles_num =  len(params["positions"])

total_tiles_shape = (np.load(params["total_tiles_shape"])*1.2).astype(int)
extra_vacuum_space =  np.load(params["extra_vacuum_space"])
print("extra_vacuum_space=",extra_vacuum_space)
clear_extend = params["clear_extend"]
half_window_lenght = params["half_window_lenght"]*1.7#1*1.7#3#1.701#2*2
window_name =  "hann"# "boxcar"#  #from scipy.signal.windows.get_window (#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html#scipy.signal.windows.get_window)

base = np.zeros((tiles_num,*total_tiles_shape))#np.zeros((tiles_num,423,423))
offsets = np.zeros((tiles_num,2),dtype=int)
pos =  []
edge_points = []
border_weight = np.zeros_like(base)
border_delete_weight = np.ones_like(base)
fig, ax = plt.subplots(2,3,figsize=(25,15),sharex=True,sharey=True)
fig.tight_layout()
debug_view = np.zeros_like(base)
debug_view2 = np.zeros_like(base)

vmin = np.inf # saving min/max for normalisations  (later plotting)
vmax = np.NINF # saving min/max for normalisations  (later plotting)

for i in range(tiles_num):
    print('index=',i,"offset ",np.load(params["tile_offset"][i]))
    pos.append(np.flip(np.load(params["positions"][i])+np.load(params["tile_offset"][i])+extra_vacuum_space,axis=1))
    tile_data = np.array(Image.open(params["recon"][i]+"/delta_ds_1.tiff"))
    tile_shape = np.load(params["tile_shape"][i])
    offsets[i] = pos[i].min(axis=0)-extra_vacuum_space#np.round(np.int64(np.load(sinc+"/tile_"+str(i)+"_offset.npy"))) #-param_extra_space_mult_scanStepSize//2))# calculate offset to position within the greater array
    print(f'offsets[{i}]',offsets[i])
    ##plt.scatter(pos[-1][:,0], pos[-1][:,1],s=1,marker='o') ### view positions
    offset_x = offsets[i][1]
    offset_y = offsets[i][0]
    print('\toffset_x=',offset_x,'\toffset_x+tile_shape[0]=',offset_x+tile_shape[0],'\toffset_y=',offset_y,'\toffset_y+tile_shape[1]=',offset_y+tile_shape[1])
    print("shape of selection into base=",offset_x-offset_x+tile_shape[0],offset_y-offset_y+tile_shape[1])
    print("tile_data.shape=",tile_data.shape)
    print("base[i].shape=",base[i].shape)
    base[i][offset_x:offset_x+tile_shape[0],offset_y:offset_y+tile_shape[1]] = tile_data[0:tile_shape[0],0:tile_shape[1]]#position data within the greater array
    vmin = np.min([vmin,np.nanmin(tile_data[0:tile_shape[0],0:tile_shape[1]])]) # saving min/max for normalisations  (later plotting)
    vmax = np.max([vmax,np.nanmax(tile_data[0:tile_shape[0],0:tile_shape[1]])]) # saving min/max for normalisations  (later plotting)

for i in range(tiles_num):
    # We want to find the corners of the non-overlaping part of the data. We got the position that include overlap and we have _slice_no_overlap.npy that specifies in 2d, the np slice of the pos that corespond to that area.
    print('shape of pos: ',pos[i].shape)
    print('reshape of pos: ', np.load(params["tile_no_overlab_sub_shape"][i]))
    tile_no_overlap_shape = np.load(params["tile_no_overlab_sub_shape"][i])
    slice_no_overlap = np.load(params["slice_no_overlap"][i])
    print("slice_no_overlap",np.load(params["slice_no_overlap"][i]))
    print('tile_no_overlap_shape',tile_no_overlap_shape)
    pos_no_overlap = (pos[i].reshape(tile_no_overlap_shape)[slice_no_overlap[0]:slice_no_overlap[1],slice_no_overlap[2]:slice_no_overlap[3]])
    pos_no_overlap = pos_no_overlap.reshape(pos_no_overlap.shape[0]*pos_no_overlap.shape[1],*pos_no_overlap.shape[2:])
    print('rereshape of pos: ',pos_no_overlap.shape)

    index_min = pos_no_overlap.argmin(axis=0)
    print(index_min,'\n',pos_no_overlap[index_min],'\n','\n')
    ax[1][0].scatter(pos_no_overlap[index_min][0][0],pos_no_overlap[index_min][0][1])
    ax[1][0].scatter(pos_no_overlap[index_min][1][0],pos_no_overlap[index_min][1][1])
    index_max = pos_no_overlap.argmax(axis=0)
    print(index_max,'\n',pos_no_overlap[index_max],'\n','\n')
    ax[1][0].scatter(pos_no_overlap[index_max][0][0],pos_no_overlap[index_max][0][1])
    ax[1][0].scatter(pos_no_overlap[index_max][1][0],pos_no_overlap[index_max][1][1])
    edge_points.append(np.array([*pos_no_overlap[index_min],*pos_no_overlap[index_max]])) #[[x1,y1],[x2,y2]...] save edgepoints
    print(edge_points[i])
    ### for full datasets


    index_min = pos[i].argmin(axis=0)
    print(index_min,'\n',pos[i][index_min],'\n','\n')
    ax[1][0].scatter(pos[i][index_min][0][0],pos[i][index_min][0][1])
    ax[1][0].scatter(pos[i][index_min][1][0],pos[i][index_min][1][1])
    index_max = pos[i].argmax(axis=0)
    print(index_max,'\n',pos[i][index_max],'\n','\n')
    ax[1][0].scatter(pos[i][index_max][0][0],pos[i][index_max][0][1])
    ax[1][0].scatter(pos[i][index_max][1][0],pos[i][index_max][1][1])
    xs = np.array([pos[i][index_min][0][0],pos[i][index_min][1][0],pos[i][index_max][0][0],pos[i][index_max][1][0]])
    ys = np.array([pos[i][index_min][0][1],pos[i][index_min][1][1],pos[i][index_max][0][1],pos[i][index_max][1][1]])
    polygon = Polygon(np.column_stack([xs, ys]), closed=True)
    # polygon = Polygon([[pos[i][index_min][0][0],pos[i][index_min][0][1]],[pos[i][index_min][1][0],pos[i][index_min][1][1]],[pos[i][index_max][0][0],pos[i][index_max][0][1]],[pos[i][index_max][1][0],pos[i][index_max][1][1]]], closed=True)
    patches = np.array([polygon])
    p = PatchCollection(patches, alpha=0.4)
    p.set_array(np.array([50]))
    ax[1][0].add_collection(p)
    text_pos_x = np.mean(xs) 
    text_pos_y = np.mean(ys) 
    # text_pos_x = np.mean([pos[i][index_min][0][0],pos[i][index_min][1][0],pos[i][index_max][0][0],pos[i][index_max][1][0]]) 
    # text_pos_y = np.mean([pos[i][index_min][0][1],pos[i][index_min][1][1],pos[i][index_max][0][1],pos[i][index_max][1][1]]) 
    ax[1][0].text(text_pos_x,text_pos_y,params["recon"][i].split("_")[-1], horizontalalignment='center', verticalalignment='center',color="white",alpha=0.5,fontsize=0.5*np.sqrt(2)*np.min(np.sqrt((text_pos_x-xs)**2+(text_pos_y-ys)**2)))


    ###
    # for a in np.ravel(ax):
    #     a.scatter(edge_points[i][:,0],edge_points[i][:,1])
    #     a.scatter(np.int64(np.round(edge_points[i][:,0])),np.int64(np.round(edge_points[i][:,1])))

    # base[i][mask_interior] = 0
    # base[i][np.logical_not(mask_interior)] = 0#((ndimage.convolve1d(border_weight[i],window,axis=0,mode='constant',cval=0.)+ndimage.convolve1d(border_weight[i],window,axis=1,mode='constant',cval=0.))/2/2/2/2/2)[np.logical_not(mask)]
    corners_poly_c = np.zeros((4,4),dtype=np.int64) #corners_polygon_colums
    corners_poly_r = np.zeros((4,4),dtype=np.int64) #corners_polygon_rows
    real_full_window_extent = -1
    for j in range(4):# go through edges
        print('\nedge', j)
        point = np.int64(np.round(edge_points[i][j]))
        next_point =  np.int64(np.round(edge_points[i][(j+1)%4]))
        # corners = {0:(slice(None,point[0]),slice(None,point[1])),1:(slice(point[0],point[0]),slice(point[1],point[1])),2:(slice(point[0],point[0]),slice(point[1],point[1])),3:(slice(point[0],point[0]),slice(point[1],point[1]))}##
        # sl = corners[j]
        # border_delete_weight[sl] = 0
        corners_poly_c[j][0] = point[1]
        corners_poly_r[j][0] = point[0]
        j_next_mod = (j+1)%4
        dx = edge_points[i][j_next_mod][0]-edge_points[i][j][0]
        dy = edge_points[i][j_next_mod][1]-edge_points[i][j][1]
        slope = dy/dx
        rr, cc = draw.line(*point,*next_point)
        for x,y,k in zip(cc,rr,range(len(cc))): # go orthogonal along line
            xstep = np.int64(np.round(1/slope*clear_extend)) # go orthogonal
            ystep = -np.int64(np.round(slope*clear_extend)) # go orthogonal
            ccx,rry  = [],[]
            # border_weight[i][ccx,rry][border_weight[i][ccx,rry]==0.0] = 1.0
            if(j in [1,2]):
              ccx,rry = draw.line(x,y,x-xstep,y-ystep)
              if k == 0:
                  corners_poly_c[j][1] = x-xstep
                  corners_poly_r[j][1] = y-ystep
              elif k == len(cc)-1:
                  corners_poly_c[j_next_mod][3] = x-xstep
                  corners_poly_r[j_next_mod][3] = y-ystep
            else:
              ccx,rry = draw.line(x+xstep,y+ystep,x,y)
              if k == 0:
                  corners_poly_c[j][1] = x+xstep
                  corners_poly_r[j][1] = y+ystep
              elif k == len(cc)-1:
                  corners_poly_c[j_next_mod][3] = x+xstep
                  corners_poly_r[j_next_mod][3] = y+ystep


            border_delete_weight[i][np.clip(ccx,0,border_delete_weight[i].shape[0]-1),np.clip(rry,0,border_delete_weight[i].shape[1]-1)] = 0.0

            xstep = np.int64(np.round(1/slope*half_window_lenght)) # go orthogonal
            ystep = -np.int64(np.round(slope*half_window_lenght)) # go orthogonal
            # ccx,rry  = draw.line(x+xstep,y+ystep,x-xstep,y-ystep)
            # border_weight[i][ccx,rry][border_weight[i][ccx,rry]==0.0] = 1.0
            if(j in [1,2]):
              ccx,rry = draw.line(x,y,x-xstep,y-ystep)
            else:
              ccx,rry = draw.line(x+xstep,y+ystep,x,y)
            l=border_weight[i][ccx,rry].shape[0]
            s =  slice(l//2,None)
            s_1 = slice(None,l//2)
            if(j in [1,2]):
                s = slice(None,l//2)
                s_1 =  slice(l//2,None)
            border_weight[i][ccx[s],rry[s]] = signal.windows.get_window(window_name,border_weight[i][ccx,rry].shape[0])[s_1]
            real_full_window_extent = border_weight[i][ccx,rry].shape[0] #asume all sides will have the same real extent
            #border_delete_weight[i][ccx[s],rry[s]] = 1
            # border_delete_weight[i][ccx[s][0],rry[s][0]] = .1

    win = signal.windows.get_window(window_name,real_full_window_extent)[real_full_window_extent//2:]
    for j in range(4):# go through cornes
        if j%2==1:
            corners_poly_c[j][2] = corners_poly_c[j][3]
            corners_poly_r[j][2] = corners_poly_r[j][1]
        else:
            corners_poly_c[j][2] = corners_poly_c[j][1]
            corners_poly_r[j][2] = corners_poly_r[j][3]
        cc,rr = draw.polygon(corners_poly_c[j],corners_poly_r[j])
        for c,r in zip(cc,rr):
            R = np.int64(np.round(np.sqrt((corners_poly_c[j][0]-c)**2+(corners_poly_r[j][0]-r)**2)))
            if R<len(win):
                debug_view[i][c,r] = win[R]
                border_weight[i][c,r] = win[R]#
            else:
                border_delete_weight[i][np.clip(c,0,border_delete_weight[i].shape[0]-1),np.clip(r,0,border_delete_weight[i].shape[1]-1)] = 0.0
        # cc,rr = draw.polygon_perimeter(corners_poly_c[j],corners_poly_r[j])
        # debug_view[i][cc,rr] = j+2

mask_interior_tot =  np.zeros(base[0].shape,dtype=bool)
mask_interior =  np.zeros((tiles_num,*base[0].shape),dtype=float)
for i in range(tiles_num):
    # mask_interior_tot |= draw.polygon2mask(base[i].shape,np.int64(np.round(edge_points[i])))
    # mask_interior[i] = draw.polygon2mask(base[i].shape,np.int64(np.round(edge_points[i]))).astype(float)
    mask_interior_tot |= draw.polygon2mask(base[i].shape,np.flip(np.int64(np.round(edge_points[i])),axis=1))
    mask_interior[i] = draw.polygon2mask(base[i].shape,np.flip(np.int64(np.round(edge_points[i])),axis=1)).astype(float)
mask_interior_weight =  mask_interior_tot.astype(float)

threshold_base = np.zeros_like(base)
for i in range(tiles_num):
    threshold_base[i][base[i]!=0] = 1
threshold_base = threshold_base.sum(axis=0)
total_weight = np.zeros_like(base[0]) # np.sum(border_weight,axis=0)#
for i in range(tiles_num):
    total_weight[base[i]!=0] += border_weight[i][base[i]!=0]# TODO posible use to fix aliasing issue? the error get very visible
total_weight += mask_interior_weight
total_delete_weight = np.sum(np.array(border_delete_weight),axis=0) # np.ones(base[0].shape)#
debug_view[0] = total_delete_weight
for i in range(tiles_num):
    # debug_view2[i][base[i]!=0] = 1
    # final_normaliced_weight = (border_weight[i][border_weight[i]!=0 and mask_interior[i]!=0]+mask_interior[i][border_weight[i]!=0 and mask_interior[i]!=0])/total_weight[border_weight[i]!=0 and mask_interior[i]!=0]
    # base[i][total_weight!=0] = base[i][total_weight!=0]*final_normaliced_weight #total_weight!=0 is used to avoid division by 0
    final_normaliced_weight = (border_weight[i][total_weight!=0]+mask_interior[i][total_weight!=0])/total_weight[total_weight!=0]



    base[i][total_weight!=0] = base[i][total_weight!=0]*final_normaliced_weight #total_weight!=0 is used to avoid division by 0
    base[i][border_weight[i]==0] *= border_delete_weight[i][border_weight[i]==0]



    # # final_normaliced_weight = (border_weight[i][base[i]!=0]+mask_interior[i][base[i]!=0]+1)/total_weight[base[i]!=0]
    # # base[i][base[i]!=0] = base[i][base[i]!=0]*final_normaliced_weight #base[i]!=0 is used to avoid division by 0

    # base[i] = border_weight[i]

    debug_view2[i] = border_weight[i]
    # # debug_view2[i][total_weight!=0] =  final_normaliced_weight
    # debug_view[i][border_weight[i]==0] = border_delete_weight[i][border_weight[i]==0]
    # debug_view2[i][total_weight!=0] = final_normaliced_weight#(border_weight[i][total_weight!=0]+mask_interior[i][total_weight!=0])/total_weight[total_weight!=0]#border_delete_weight[i]#
    # debug_view2[i] = border_delete_weight[i]#
    # base[i] = border_weight[i]
    # base[i][border_weight[i]==0] *= border_delete_weight[i][border_weight[i]==0]
    # base[i] += border_weight[i]
    #
    # print()
ax00 = ax[0][0].imshow(base.sum(axis=0),cmap='gray',vmin=vmin,vmax=vmax,interpolation='none')
fig.colorbar(ax00,ax=ax[0][0])
ax[0][0].set_title('sum')

ax01 = ax[0][1].imshow(base[0],cmap='gray',vmin=vmin,vmax=vmax,interpolation='none')
fig.colorbar(ax01,ax=ax[0][1])
ax[0][1].set_title('one tile')
ax02 = ax[0][2].imshow(debug_view[0],cmap='gray',interpolation='none')
fig.colorbar(ax02,ax=ax[0][2])
ax[0][2].set_title('total_delete_weight (debug_view)')
# ax[1][1].imshow(base[1],cmap='gray',interpolation='none')
# ax[1][1].set_title('one')
# ax[0][1].imshow(base[1],cmap='gray',interpolation='none')
ax10 = ax[1][0].imshow(base.sum(axis=0),cmap='gray',vmin=vmin,vmax=vmax,interpolation='none')
fig.colorbar(ax10,ax=ax[1][0])
ax[1][0].set_title('Overlap vis.')
ax11 = ax[1][1].imshow(debug_view2[0],cmap='gray',interpolation='none')
fig.colorbar(ax11,ax=ax[1][1])
ax[1][1].set_title('border_weight (debug_view2)')
ax12 = ax[1][2].imshow(total_weight,cmap='gray',interpolation='none')
fig.colorbar(ax12,ax=ax[1][2])
ax[1][2].set_title('total_weight')#
# plt.plot(offsets[:,1],offsets[:,0],'x')
# print(np.min(base[base!=0.0]),np.max(base[0]))
# ax[0][0].set_xlim(240,320)
# ax[0][0].set_ylim(470,580)
fig.savefig("debug_overview.pdf",bbox_inches='tight')

base_sum = base.sum(axis=0)
#tiff
im = Image.fromarray(base_sum)
im.save("recon.tiff",format="TIFF")#,compression="tiff_lzw"

#png
fig_out, ax_out = plt.subplots(1,1,figsize=(base.shape[1]*px,base.shape[2]*px))
ax_out_ = ax_out.imshow(base_sum,cmap='gray',interpolation='none')
plt.imsave("recon_for_vis.png",base_sum,format='png',cmap='gray',vmin=vmin,vmax=vmax)
fig_out.colorbar(ax_out_,ax=ax_out)
fig_out.savefig("recon_colorbar.pdf",bbox_inches='tight')

#png with normalisation taken from the raw tiles
fig_out, ax_out = plt.subplots(1,1,figsize=(base.shape[1]*px,base.shape[2]*px))
ax_out_ = ax_out.imshow(base_sum,cmap='gray',vmin=vmin,vmax=vmax)
fig_out.colorbar(ax_out_,ax=ax_out)
plt.imsave("recon_without_vacuum_norm_for_vis.png",base_sum,format='png',cmap='gray',vmin=vmin,vmax=vmax)
fig_out.savefig("recon_without_vacuum_norm_colorbar.pdf",bbox_inches='tight')
#save vmin/vmax
np.savetxt("vmin_vmax.txt",[vmin,vmax])

# total_tiles_shape = np.load(params["total_tiles_shape"])
# print("total_tiles_shape: ", total_tiles_shape)
# print("recon: ", params["recon"])
# tile_temp = dxchange.read_tiff(params["recon"][0]+"/delta_ds_1.tiff") # Hack, real value should be passes
# print("FIX ME! temp = ...")
# print("tile_temp.shape",tile_temp.shape)
# probe_temp = dxchange.read_tiff(params["recon"][0]+"/probe_mag_ds_1.tiff")
# print("one probe.shape",probe_temp.shape)
# probe_mag = np.empty((len(params["recon"]),*probe_temp.shape),dtype=probe_temp.dtype)
# probe_phase = np.empty((len(params["recon"]),*probe_temp.shape),dtype=probe_temp.dtype)
# tiler = Tiler(data_shape=(*total_tiles_shape,tile_temp.shape[-1]),
#               tile_shape=tile_temp.shape)
# merger_beta = Merger(tiler)
# merger_delta = Merger(tiler)

# for t in params["recon"]:
#     tile_id = int(t[-1])
#     merger_beta.add(tile_id, dxchange.read_tiff(t+"/beta_ds_1.tiff"))
#     merger_delta.add(tile_id, dxchange.read_tiff(t+"/delta_ds_1.tiff"))
#     probe_mag[tile_id] = dxchange.read_tiff(t+"/probe_mag_ds_1.tiff")[0]
#     probe_phase[tile_id] = dxchange.read_tiff(t+"/probe_phase_ds_1.tiff")[0]

# final_beta = merger_beta.merge(unpad=True)
# final_delta = merger_delta.merge(unpad=True)
# dxchange.write_tiff(final_beta,"beta_ds_joined.tiff")
# dxchange.write_tiff(final_delta,"delta_ds_joined.tiff")
# dxchange.write_tiff(probe_mag,"probe_mag_ds_joined.tiff")
# dxchange.write_tiff(probe_phase,"probe_phase_ds_joined.tiff")