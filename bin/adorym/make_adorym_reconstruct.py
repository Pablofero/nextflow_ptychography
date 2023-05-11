import sys
import numpy as np

params={}
fname = 'something_went_wrong' # some default to catch problems
obj_shape_ = [0,0]
probe_size_ = [1e6,1e6] # some default to catch problems
pos_len = 0
for ste_name,set_conf in zip(sys.argv[1:-1:2],sys.argv[2::2]):
	# print(ste_name,set_conf)
	if ste_name =='--fname':
		fname = set_conf 
		params[ste_name.lstrip('--')] = "\'"+set_conf+"\'"
	elif ste_name =='--probe_pos':
		params[ste_name.lstrip('--')] = "np.load(\'"+set_conf+"\')" # np.flipud(np.fliplr(
		pos_len = np.load(set_conf).shape[0]
		# params[ste_name.lstrip('--')] = "np.load(\'"+set_conf+"\')"
		# params[ste_name.lstrip('--')] = "np.load(\'"+"/testpool/ops/pablofernandezrobledo/Workflows/nextflow_preprocessing/data_anton/positions_set_y_reversed_x_reversed.npy"+"\')"
	elif ste_name == "--minibatch_size":
		win = 1
		for i in range(int(set_conf),0,-1):
			if pos_len%i==0:
				win = i
				break
		params[ste_name.lstrip('--')] = win
	elif(ste_name=='--beamstop'):
		if set_conf != 'None':
			params[ste_name.lstrip('--')] = "np.load(\'"+set_conf+"\')"
	elif ste_name =='--probe_size':
		probe_size_ = np.load(set_conf)
		params[ste_name.lstrip('--')] = tuple(probe_size_)
	elif ste_name =='--shape':
		obj_shape_ = np.load(set_conf).astype(int)
	elif(ste_name=='--obj_size'):
			# params[ste_name.lstrip('--')] = dxchange.read_tiff(fname).shape, params[ste_name.lstrip('--')] #"np.load(\'"+set_conf+"\')"
			get_z = np.fromstring(set_conf.strip('(').lstrip(')'),dtype=int,sep=',')# from "(x,y,1)" to get the array [x,y,z]
			params[ste_name.lstrip('--')] = (*obj_shape_, get_z[-1])# dxchange.read_tiff(fname).shape, params[ste_name.lstrip('--')] #"np.load(\'"+set_conf+"\')"
	elif ste_name =='--output_folder':
		fname = set_conf 
		params[ste_name.lstrip('--')] = "\'"+fname+"\'" # put some ' so it gets read as a string
	else:
		if(set_conf=='false' or set_conf=='true'):
			params[ste_name.lstrip('--')] = set_conf.title()
		else:
			params[ste_name.lstrip('--')] = set_conf

with open('adorym_recostruct.py','w') as f:
	f.write("""from adorym.ptychography import reconstruct_ptychography
import adorym
import numpy as np
import dxchange
import datetime
import argparse
import os

""")
	f.write('params = {\n')
	first = True
	for elem in params:
		if elem == 'cfg':
			continue
		if not first:
			f.write(',\n')
		else:
			first = False
		f.write('\t\''+str(elem)+'\':'+str(params[elem]))
	f.write('\n}')
	f.write('\n')
	f.write('\n')

	f.write('reconstruct_ptychography(**params)')#