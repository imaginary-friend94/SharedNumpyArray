### Shared Numpy Array cross-platform [python 3]
![test workflow](https://github.com/imaginary-friend94/SharedNumpyArray/actions/workflows/tests.yml/badge.svg)
[![Release](https://img.shields.io/github/v/release/imaginary-friend94/SharedNumpyArray)](https://github.com/imaginary-friend94/SharedNumpyArray/releases)

*Share numpy arrays between processes*
<br/>

![image](https://github.com/imaginary-friend94/SharedNumpyArray/assets/15076754/a37651bd-844c-45e9-b359-012be75ca69f)

*FOR OS WINDOWS:* you should use slash as first symbol of shared memory name. for ex: /shm_mem_npy

**how use shared array:**
```python
#process 1

import numpysharedarray as nps
import numpy as np

arr = np.zeros((15,5))

#this function create new array and copy in shared memory
nps.create_mem_sh("shm_mem_npy", arr)
#after create let's attach to shared array
array_attached = nps.attach_mem_sh("shm_mem_npy")
#you can fulfill any numpy operation
array_attached[:3,:1] = 1
array_attached[:3,0] = np.sin(array_attached[:3,0])
array_attached[:1,1:] += 12

#process 2
import numpysharedarray as nps
import numpy as np

#shared array has already been created so just attach
array_attached = nps.attach_mem_sh("shm_mem_npy")
# array([[ 0.84147098, 12.        , 12.        , 12.        , 12.        ],
#        [ 0.84147098,  0.        ,  0.        ,  0.        ,  0.        ],
#        [ 0.84147098,  0.        ,  0.        ,  0.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])
print(array_attached)
```


**how use shared array and mutex:**
```python
import numpysharedarray as nps
import numpy as np

from multiprocessing import Pool

def mutex_test(num):
    mutx = nps.open_mutex("mutex_name")
    cntr = nps.attach_mem_sh("counter_name")
    #we protect the array using a mutex
    nps.capture_mutex(mutx)
    cntr[0][0] += 1
    nps.release_mutex(mutx)

counter_array = np.zeros((1,1))
nps.create_mem_sh("counter_name", counter_array)
mutx = nps.create_mutex("mutex_name")

#adding to 5 threads
p = Pool(5)
p.map(mutex_test, range(5000))
attach_array = nps.attach_mem_sh("counter_name")
print(attach_array)
```
