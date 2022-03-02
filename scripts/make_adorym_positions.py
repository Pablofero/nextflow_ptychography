import numpy as np

from jsonargparse import ArgumentParser, ActionConfigFile
from jsonargparse.typing import path_type
Path_nocheck = path_type('rw', docstring='str pointing to a folder', skip_check=True)# for directories
Path_f_nocheck =  path_type('frw', docstring='str pointing to a file', skip_check=True) # for files
# if you want to check use Path_dw or Path_fr from jsonargparse.typing or create custom https://jsonargparse.readthedocs.io/en/stable/index.html#parsing-paths
parser = ArgumentParser(parse_as_dict=True)
parser.add_argument('--cfg', action=ActionConfigFile)
parser.add_argument("--N_scan_x", type=int, help="", default=50)
parser.add_argument("--N_scan_y", type=int, help="", default=50)
parser.add_argument("--scan_Step_Size_x", type=float, help="", default=.2)
parser.add_argument("--scan_Step_Size_y", type=float, help="", default=.2)
parser.add_argument("--rot_ang_str", type=str, help="evaluates the string, use with care!", default="np.pi/180*(333.68)")
params = parser.parse_args()



N_scan_x=params['N_scan_x']
N_scan_y=params['N_scan_y']
scanStepSize_x=params['scan_Step_Size_x']
scanStepSize_y=params['scan_Step_Size_y']
rot_ang = eval(params['rot_ang_str']);rot_ang_str = params['rot_ang_str'].replace('/','⁄')


out_name = 'beam_pos_'+rot_ang_str+'='+str(rot_ang)+'_'
out_name += 'Scan'+str(N_scan_x)+'x'+str(N_scan_y)+'_'
out_name += 'StepsSize'+str(scanStepSize_x)+'x'+str(scanStepSize_y)+'.npy'

ppx = np.linspace(-np.floor(N_scan_x / 2), np.ceil(N_scan_x / 2) - 1,
N_scan_x) * scanStepSize_x
ppy = np.linspace(-np.floor(N_scan_y / 2), np.ceil(N_scan_y / 2) - 1,
N_scan_y) * scanStepSize_y
[ppX, ppY] = np.meshgrid(ppx, ppy)

ppY_rot = ppX * np.cos(rot_ang) + ppY * (-np.sin(rot_ang))
ppX_rot = ppX * np.sin(rot_ang) + ppY * np.cos(rot_ang)

#print(ppY_rot + 140)
#print(ppX_rot + 140)


A=[]
B=[]
C=[]
for i in range(0,N_scan_x):
         for j in range(0,N_scan_y):
                 a = ppX_rot[j,i] * 10 ** (-10)
                 b = ppY_rot[j,i] * 10 ** (-10)
                 A=np.append(A,a)
                 B=np.append(B,b)

from matplotlib import pyplot as plt
plt.scatter(A,B)
plt.xlim(-10**(-9),10**(-9))
plt.ylim(-10**(-9),10**(-9))
plt.savefig('beam_pos.png')
#plt.show()


#Permutate
tempA = []
tempB = []
for j in range(N_scan_y):
         for i in range(N_scan_x):
                 i_ = i * N_scan_y + j
                 tempA = np.append(tempA,A[i_])
                 tempB = np.append(tempB,B[i_])

Z=np.ones(B.size)
#M=np.zeros((B.size,3))
M=np.zeros((B.size,2))

#tempB += np.random.rand(64*64) * (0.86*2/3) * 10 ** (-10) - (0.43*2/3) * 10 ** (-10)
#tempA += np.random.rand(64*64) * (0.86*2/3) * 10 ** (-10) - (0.43*2/3) * 10 ** (-10)

for i in range(B.size):
         Z[i] = 1111
         #M[i]=np.hstack((Z[i],tempB[i],tempA[i]))
         M[i]=np.hstack((tempA[i],tempB[i]))

M = M/1.027e-11 #+ 106.5
# M -= np.min(M)
# print(M)
#np.savetxt('/home/marcel/Desktop/Positions.txt',M,fmt='%1.4e',newline='\n')
np.save(out_name,M)#og: f'/Users/Tom/mnt/ops/tompekin/tbg/adorym_results/beam_pos_{rot_ang*180/np.pi:.2f}_50x50.npy'       