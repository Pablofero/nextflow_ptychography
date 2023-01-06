import sys
import numpy as np

params={}
fname = 'something_went_wrong' # some default to catch problems
shape_ = [0,0]
probe_size_ = [1e6,1e6] # some default to catch problems
for ste_name,set_conf in zip(sys.argv[1:-1:2],sys.argv[2::2]):
	# print(ste_name,set_conf)
	if ste_name =='--fname':
		fname = set_conf 
		params[ste_name.lstrip('--')] = "\'"+set_conf+"\'"
	elif ste_name =='--probe_pos':
		params[ste_name.lstrip('--')] = "np.load(\'"+set_conf+"\')" # np.flipud(np.fliplr(
		# params[ste_name.lstrip('--')] = "np.load(\'"+set_conf+"\')"
		# params[ste_name.lstrip('--')] = "np.load(\'"+"/testpool/ops/pablofernandezrobledo/Workflows/nextflow_preprocessing/data_anton/positions_set_y_reversed_x_reversed.npy"+"\')"
	elif(ste_name=='--beamstop'):
		if set_conf != 'None':
			params[ste_name.lstrip('--')] = "np.load(\'"+set_conf+"\')"
	elif ste_name =='--probe_size':
		probe_size_ = np.load(set_conf)
		params[ste_name.lstrip('--')] = tuple(probe_size_)
	elif ste_name =='--shape':
		shape_ = np.load(set_conf)
		shape_ = (np.load(set_conf)*1.1).astype(np.int)
	elif(ste_name=='--obj_size'):
			# params[ste_name.lstrip('--')] = dxchange.read_tiff(fname).shape, params[ste_name.lstrip('--')] #"np.load(\'"+set_conf+"\')"
			z = np.fromstring(set_conf.strip('(').lstrip(')'),dtype=int,sep=',')
			params[ste_name.lstrip('--')] = (*shape_, z[-1])# dxchange.read_tiff(fname).shape, params[ste_name.lstrip('--')] #"np.load(\'"+set_conf+"\')"
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