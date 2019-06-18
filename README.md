### Shared Array for Windows [python 3]

**example:**
```python
import winsharedarray as sa
import numpy as np

arr = np.zeros((5,5))

sa.set_mem_sh("shm_mem_npy", arr)
print(sa.get_mem_sh("shm_mem_npy", arr.nbytes))
```