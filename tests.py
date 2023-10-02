import numpysharedarray as sa
import numpy as np

# def test_create_and_remove():
# 	arr = np.zeros((10,10))
# 	assert sa.create_mem_sh("/base_share_array_cr", arr) == True
# 	assert sa.check_mem_sh("/base_share_array_cr") == True
# 	sa.delete_mem_sh("/base_share_array_cr")
# 	assert sa.check_mem_sh("/base_share_array_cr") == False

def test_array_int():
	arr = np.zeros((1920,1080)).astype(np.int32)
	assert sa.create_mem_sh("/base_share_array_int", arr) == True

	array_attached = sa.attach_mem_sh("/base_share_array_int")
	arr = sa.attach_mem_sh("/base_share_array_int")
	assert sa.check_mem_sh("/base_share_array_int") == True
	assert array_attached.dtype == np.dtype(np.int32)
	#shared numpy array
	arr[:50, 50:87] = 5

	assert np.all(array_attached[:50, 50:87] == 5)
	assert np.all(array_attached[60:, 20:600] == 0)
 
def test_array_float():
	arr = np.zeros((1920,1080)).astype(np.float32)
	assert sa.create_mem_sh("/base_share_array_int", arr) == True

	array_attached = sa.attach_mem_sh("/base_share_array_int")
	arr = sa.attach_mem_sh("/base_share_array_int")
	assert sa.check_mem_sh("/base_share_array_int") == True
	assert array_attached.dtype == np.dtype(np.float32)
	#shared numpy array
	arr[:50, 50:87] = 5

	assert np.all(array_attached[:50, 50:87] == 5.)
	assert np.all(array_attached[60:, 20:600] == 0.)