### Shared Numpy Array cross-platform [python 3]
![test workflow](https://github.com/imaginary-friend94/SharedNumpyArray/actions/workflows/tests.yml/badge.svg)
[![Release](https://img.shields.io/github/v/release/imaginary-friend94/SharedNumpyArray)](https://github.com/imaginary-friend94/SharedNumpyArray/releases)

*Share numpy arrays between processes*
<br/>

![image](https://github.com/imaginary-friend94/SharedNumpyArray/assets/15076754/0e845f40-2589-41e2-a89d-9172b5f6efcf)

**example:**
```python
import winsharedarray as sa
import numpy as np

arr = np.zeros((5,5))

sa.create_mem_sh("shm_mem_npy", arr)
array_attached = sa.attach_mem_sh("shm_mem_npy")
array_attached[:3,:1] = 1


