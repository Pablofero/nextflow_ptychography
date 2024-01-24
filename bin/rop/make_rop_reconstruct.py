import sys
import numpy as np
import matplotlib.pyplot as plt
import struct

params={}
beam_position = None
probe_generated = None
ObjectDim = None
random_guess_means_sigmas = np.array([0, 0, 0, 0])
for ste_name,set_conf in zip(sys.argv[1:-1:2],sys.argv[2::2]):
    if "(" in set_conf:
        Microscope_parameter = set_conf.strip("(").strip(")").split(",")
        params[ste_name.lstrip('--')] = "   "+"   ".join(Microscope_parameter)
    elif ste_name =='--ObjectDim':
        ObjectDim = np.max(np.load(set_conf)) #TODO check if this works
        params[ste_name.lstrip('--')] = ObjectDim
    elif ste_name == "--ProbeDim":
        ProbeDim = np.max(np.load(set_conf)) #TODO check if this works
        params[ste_name.lstrip('--')] = ProbeDim
    elif ste_name == "--PixelSize":
        PixelSize = np.max(np.load(set_conf))
        params[ste_name.lstrip('--')] = PixelSize
    elif ste_name == "--CBEDDim":
        params[ste_name.lstrip('--')]  = np.max(np.load(set_conf))
    elif ste_name == "--beam_position":
        beam_position = np.load(set_conf)
    elif ste_name == "--probe_generated":
        probe_generated = np.load(set_conf)
    elif ste_name == "--random_guess_means_sigmas":
        random_guess_means_sigmas = np.fromstring(str(set_conf).replace('[','').replace(']',''),dtype=float,sep=",")
    else:
    	params[ste_name.lstrip('--')] = set_conf
	
# beam_position = beam_position.T
# pos_size = int(beam_position.shape[1] / 2)
# #Rescale beam_position
# with: Center beam_position with top_left static
top_left = beam_position.min(axis=0)
beam_position -= top_left
beam_position *= params["PixelSize"]
beam_position += params["PixelSize"]*top_left
beam_position -= params["PixelSize"]*ProbeDim/2
beam_position -= params["PixelSize"]*np.array([1,35])
#Combine all together
# beam_position_M = np.vstack((beam_position[1][:pos_size],beam_position[0][:pos_size]))
beam_position_M = np.vstack((beam_position[1][:],beam_position[0][:]))
beam_position_M = beam_position_M.T

with open('Params.cnf','w') as f:
    f.write("""Parameter file for ROP
Let Comments be precede d by '#'

""")
    for elem in params:
        # if elem == 'cfg':
        # 	continue
        f.write(str(elem)+': '+str(params[elem])+'\n')

    f.write("""\n###Arbitrary beam position specification \n#x- and y- coordinate [m].\n""")
    for i in beam_position: # beam_position_M
        f.write('beam_position:   '+str(i[1])+' '+str(i[0])+'\n')

with open("Probe_re.bin","wb") as file_re, open("Probe_im.bin","wb") as file_im:
    N = probe_generated.shape
    data_char = "f"
    
    probe_re = np.real(probe_generated)
    probe_re = np.ravel(probe_re)
    probe_re = struct.pack(np.prod(N) * data_char, *probe_re)
    file_re.write(probe_re)
    
    probe_im = np.imag(probe_generated)
    probe_im = np.ravel(probe_im)
    probe_im = struct.pack(np.prod(N) * data_char, *probe_im)
    file_im.write(probe_im)


with open("Object_re.bin","wb") as file_re, open("Object_im.bin","wb") as file_im:
    object_re = np.random.normal(size=2*(ObjectDim,), loc=random_guess_means_sigmas[0], scale=random_guess_means_sigmas[2])
    object_im = np.random.normal(size=2*(ObjectDim,), loc=random_guess_means_sigmas[1], scale=random_guess_means_sigmas[3])
    
    N = object_re.shape
    data_char = "f"
    
    object_re = np.ravel(object_re)
    object_re = struct.pack(np.prod(N) * data_char, *object_re)
    file_re.write(object_re)
    
    object_im = np.ravel(object_im)
    object_im = struct.pack(np.prod(N) * data_char, *object_im)
    file_im.write(object_im)