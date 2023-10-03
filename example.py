from multiprocessing import Pool
import numpysharedarray as nps
import numpy as np

def mutex_test(num):
	mutx = nps.open_mutex("sem_cntr_ex")
	cntr = nps.attach_mem_sh("/shm_mem_npy_counter")
	for ix in range(num):
		nps.capture_mutex(mutx)
		cntr[0][0] += 1
		nps.release_mutex(mutx)

if __name__ == '__main__':
	print(dir(nps))

	## TEST 1
	print("TEST 1")

	arr = np.zeros((5,5))

	print(nps.check_mem_sh("/shm_mem_npy_test_1f"))
	nps.create_mem_sh("/shm_mem_npy_test_1f", arr)

	try:
		array_attached = nps.attach_mem_sh("/shm_mem_npy_test_12f")
	except:
		array_attached = nps.attach_mem_sh("/shm_mem_npy_test_1f")	

	array_attached[:3,:1] = 1
	print(array_attached)
	print(nps.delete_mem_sh("/shm_mem_npy_test_1f"))
	print(nps.check_mem_sh("/shm_mem_npy_test_1f"))

	## TEST 2
	print("TEST 2")

	arr = np.zeros((1081, 1920, 24))

	nps.create_mem_sh("/shm_mem_npy_test_2", arr)

	try:
		array_attached = nps.attach_mem_sh("/shm_mem_npy_test_12")
	except:
		array_attached = nps.attach_mem_sh("/shm_mem_npy_test_2")	

	array_attached[:30,:300] = 88.
	print(arr.mean())
	print(array_attached[:,:,:1].mean())
	nps.delete_mem_sh("/shm_mem_npy_test_2")
	## TEST 3
	print("TEST 3")

	arr = np.zeros((4,4)).astype(np.float64)
	nps.create_mem_sh("/shm_mem_npy_test_3", arr)
	array_attached = nps.attach_mem_sh("/shm_mem_npy_test_3")
	# do something
	array_attached = nps.attach_mem_sh("/shm_mem_npy_test_3")
	#import time
	#time.sleep(10)
	#print(nps.delete_mem_sh("/shm_mem_npy_test_3"))

	array_attached = nps.attach_mem_sh("/shm_mem_npy_test_3")	

	array_attached[:3,:1] = 1
	print(array_attached)

	print(
		nps.check_mem_sh("/shm_mem_npy_test_3"),
		nps.check_mem_sh("/shm_mem_npy_test_55")
	)
	nps.delete_mem_sh("/shm_mem_npy_test_3")
	nps.create_mem_sh("/shm_mem_npy_test_55", arr)
	print(
		nps.check_mem_sh("/shm_mem_npy_test_3"),
		nps.check_mem_sh("/shm_mem_npy_test_55")
	)
	nps.delete_mem_sh("/shm_mem_npy_test_55")
	## TEST 4
	print("TEST 4")

	mutx = nps.create_mutex("sem_cntr_ex")
	nps.release_mutex(mutx)

	arr = np.zeros((1,1))
	nps.create_mem_sh("/shm_mem_npy_counter", arr)

	p = Pool(5)
	p.map(mutex_test, list(range(1000)))

	cntr = nps.attach_mem_sh("/shm_mem_npy_counter")
	print(cntr[0][0]) # 499500
	nps.delete_mem_sh("/shm_mem_npy_counter")
	print("GetLastError() = ", nps.get_last_error())

