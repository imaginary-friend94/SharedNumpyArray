import winsharedarray as sa
import numpy as np

arr = np.zeros((5,5))

sa.create_mem_sh("shm_mem_npy", arr)

try:
	array_attached = sa.attach_mem_sh("shm_mem_npy2", arr.nbytes)
except RuntimeError:
	array_attached = sa.attach_mem_sh("shm_mem_npy", arr.nbytes)	

array_attached[:3,:1] = 1
