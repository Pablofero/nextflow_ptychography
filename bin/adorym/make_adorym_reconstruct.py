import sys

params={}
for ste_name,set_conf in zip(sys.argv[1:-1:2],sys.argv[2::2]):
	# print(ste_name,set_conf)
	if ste_name =='--fname':
		params[ste_name.lstrip('--')] = "\'"+set_conf+"\'"
	elif ste_name =='--probe_pos':
		params[ste_name.lstrip('--')] = "np.flipud(np.fliplr(np.load(\'"+set_conf+"\')))"
	elif(ste_name=='--beamstop'):
		params[ste_name.lstrip('--')] = "np.load(\'"+set_conf+"\')"
	else:
		if(set_conf=='false' or set_conf=='true'):
			params[ste_name.lstrip('--')] = set_conf.title()
		else:
			params[ste_name.lstrip('--')] = set_conf

with open('adorym_recostruct.py','w') as f:
	f.write("""from adorym.ptychography import reconstruct_ptychography
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