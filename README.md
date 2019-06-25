### Shared Array for Windows [python 3]
*Share numpy arrays between processes*
<br/>
**example:**
```python
import winsharedarray as sa
import numpy as np

arr = np.zeros((5,5))

sa.create_mem_sh("shm_mem_npy", arr)
array_attached = sa.attach_mem_sh("shm_mem_npy", arr.nbytes)
array_attached[:3,:1] = 1


