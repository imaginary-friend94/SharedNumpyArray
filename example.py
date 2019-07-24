import winsharedarray as sa
import numpy as np

print(dir(sa))

## TEST 1

arr = np.zeros((5,5))

sa.create_mem_sh("shm_mem_npy_test_1", arr)

try:
	array_attached = sa.attach_mem_sh("shm_mem_npy_test_12", arr.nbytes)
except RuntimeError:
	array_attached = sa.attach_mem_sh("shm_mem_npy_test_1", arr.nbytes)	

array_attached[:3,:1] = 1
print(array_attached)

## TEST 2

arr = np.zeros((4,4)).astype(np.float64)
sa.create_mem_sh("shm_mem_npy_test_2", arr)
array_attached = sa.attach_mem_sh("shm_mem_npy_test_2", arr.nbytes)
# do something
array_attached = sa.attach_mem_sh("shm_mem_npy_test_2", arr.nbytes)

print(sa.delete_mem_sh("shm_mem_npy_test_2", arr.nbytes))

array_attached = sa.attach_mem_sh("shm_mem_npy_test_2", arr.nbytes)	

array_attached[:3,:1] = 1
print(array_attached)

## TEST 3

mutx = sa.create_mutex("sem_test_1")
print(sa.release_mutex(mutx))
print(sa.open_mutex("sem_test_1"))