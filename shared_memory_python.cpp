#if defined(_WIN64) || defined(_WIN32) || defined(__CYGWIN__)
#define WIN
#include <Windows.h>
#include <winbase.h>
#include <sddl.h>
#pragma comment(lib, "advapi32.lib")
#elif defined(__linux__)
#define LINUX
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <semaphore.h>

struct sem_wrapper
{
	sem_t * sem;
	bool is_locked;
};

#endif

#include "Python.h"
#include "numpy/arrayobject.h"
#include <iostream>
#include <cstdio>
#include <cstring>

#define WIN_SMEM "WINDOWS SHARED MEMORY"
#define ARRAY_STRUCT_SIZE sizeof(PyArrayObject)
#define ARRAY_FULL_SIZE(arr) (size_data_array(arr) + sizeof(int) + arr->nd * sizeof(npy_intp) * 3 + sizeof(int))


#if defined(WIN)

BOOL CreateMyDACL(SECURITY_ATTRIBUTES * pSA)
{
     TCHAR * szSD = TEXT("D:")       // Discretionary ACL
        TEXT("(D;OICI;GA;;;BG)")     // Deny access to 
                                     // built-in guests
        TEXT("(D;OICI;GA;;;AN)")     // Deny access to 
                                     // anonymous logon
        TEXT("(A;OICI;GRGWGX;;;AU)") // Allow 
                                     // read/write/execute 
                                     // to authenticated 
                                     // users
        TEXT("(A;OICI;GA;;;BA)");    // Allow full control 
                                     // to administrators

    if (NULL == pSA)
        return FALSE;

     return ConvertStringSecurityDescriptorToSecurityDescriptor(
                szSD,
                SDDL_REVISION_1,
                &(pSA->lpSecurityDescriptor),
                NULL);
}
#endif

/*
 * copy_from_pointer_array
 * the method returns the number of bytes copied
 */
template <typename ObjectType>
std::size_t copy_from_pointer_array(ObjectType * buffer_dist, ObjectType * buffer_src, size_t len) {
	for (int i = 0; i < len; i++) {
		*buffer_dist = *buffer_src;
		buffer_dist++;
		buffer_src++;
	}
	return len * sizeof(ObjectType);
}

std::size_t size_data_array(PyArrayObject *arr) {
	if (arr->nd == 0)
		return 0;

	std::size_t size = 1;
	for (int i = 0; i < arr->nd; ++i) {
		size *= (int) PyArray_DIM(arr, i);
	}
	size *= PyArray_ITEMSIZE(arr);
	return size;
}

void copy_from_numpy_array_to_buffer(PyArrayObject * array, char * buffer) {
	char * current_pointer = buffer;
	*((int *) current_pointer) = array->nd;
	current_pointer += sizeof(int);
	// dimensions copy
	current_pointer += copy_from_pointer_array(
		(npy_intp * ) current_pointer, 
		(npy_intp * ) array->dimensions, 
		array->nd
	);
	// strides copy
	current_pointer += copy_from_pointer_array(
		(npy_intp * ) current_pointer, 
		(npy_intp * ) array->strides, 
		array->nd
	);
	*((int *) current_pointer) = array->descr->type_num; 
	current_pointer += sizeof(int);

	size_t size_data = size_data_array(array);
	/* Copy data from heap to mmap memory */
	std::memcpy((char *) (current_pointer), (char *) array->data, size_data);
}

PyArrayObject * copy_from_buffer_to_numpy_array(char * buffer) {
	char * current_pointer = buffer;
	int nd = *((int *) current_pointer);
	current_pointer += sizeof(int);
	npy_intp * dims = new npy_intp[nd];

	current_pointer += copy_from_pointer_array(
		(npy_intp * ) dims, 
		(npy_intp * ) current_pointer, 
		nd
	);
	npy_intp * strides = new npy_intp[nd];
	current_pointer += copy_from_pointer_array(
		(npy_intp * ) strides, 
		(npy_intp * ) current_pointer, 
		nd
	);
	int type_num = *((int *) current_pointer);
	current_pointer += sizeof(int);

	PyArrayObject * array = (PyArrayObject *) PyArray_SimpleNewFromData(nd, 
		dims, 
		type_num, 
		(void *) current_pointer
	);
	return array;
}


/*
 * Create a buffer in shared memory
 */
char * create_shared_memory(char * string_shm, int max_buffer_size) {
	bool error_open_file_flag = false;
#if defined(WIN)
	SECURITY_ATTRIBUTES sa;
	if (!CreateMyDACL(&sa))
    {
         // Error encountered; generate message and exit.
		PyErr_SetString(PyExc_RuntimeError, "create_mutex: failed CreateMyDACL");
		return nullptr;
    }
	HANDLE hMapFile;
	hMapFile = CreateFileMapping(
		INVALID_HANDLE_VALUE,
		&sa,
		PAGE_READWRITE,
		0,
		max_buffer_size,
		string_shm);
	if (hMapFile == NULL) error_open_file_flag = true;
#elif defined(LINUX)
	int hMapFile = shm_open(string_shm, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
	if (hMapFile < 0){ 
		error_open_file_flag = true; 
	} else {
		if (ftruncate(hMapFile, max_buffer_size) == -1) error_open_file_flag = true;
	}
#endif

	if (error_open_file_flag) {
		PyErr_SetString(PyExc_RuntimeError, "create file is failed");
		return nullptr;
	}

#if defined(WIN)
	char * pBuf = (char *) MapViewOfFile(hMapFile,
                        FILE_MAP_ALL_ACCESS,
                        0,
                        0,
                        max_buffer_size);
#elif defined(LINUX)
	char * pBuf = (char *) mmap(NULL, max_buffer_size, PROT_WRITE | PROT_READ, MAP_SHARED, hMapFile, 0);
#endif

	if (pBuf == nullptr) {
		PyErr_SetString(PyExc_RuntimeError, "memory not allocated");
		return nullptr;
	}

	return pBuf;
}
/*
 * Del a buffer in shared memory
 */
bool delete_shared_memory(char * string_shm) {
#if defined(WIN)
	return true;
#elif defined(LINUX)
	if (shm_unlink(string_shm) == 0) return true;
#endif
}

/*
 * Attach a buffer in shared memory
 */
char * attach_shared_memory(char * string_shm) {
	bool error_open_file_flag = false;
#if defined(WIN)
	HANDLE hMapFile = OpenFileMapping(
		FILE_MAP_ALL_ACCESS,
		FALSE,
		string_shm); 
	if (hMapFile == NULL) error_open_file_flag = true;
#elif defined(LINUX)
	int hMapFile = shm_open(string_shm, O_RDWR, 0);
	if (hMapFile == -1) error_open_file_flag = true;
#endif

	if (error_open_file_flag) {
		PyErr_SetString(PyExc_RuntimeError, "memory not attached");
		return nullptr;
	}

#if defined(WIN)
	char * pBuf = (char *) MapViewOfFile(hMapFile,
                        FILE_MAP_ALL_ACCESS,
                        0,
                        0,
                        sizeof(size_t));
#elif defined(LINUX)
	char * pBuf = (char *) mmap(0, sizeof(size_t), PROT_WRITE | PROT_READ, MAP_SHARED, hMapFile, 0);
#endif

	size_t full_array_size = *((size_t *) pBuf);

#if defined(WIN)
	UnmapViewOfFile((LPCVOID) pBuf);
	pBuf = (char *) MapViewOfFile(hMapFile,
                        FILE_MAP_ALL_ACCESS,
                        0,
                        0,
                        full_array_size);
#elif defined(LINUX)
	munmap(pBuf, sizeof(size_t));
	pBuf = (char *) mmap(0, full_array_size, PROT_WRITE | PROT_READ, MAP_SHARED, hMapFile, 0);
#endif

	pBuf += sizeof(size_t);

	if (pBuf == nullptr) {
		PyErr_SetString(PyExc_RuntimeError, "memory not attached");
		return nullptr;
	}

	return pBuf;
}

static PyObject *
check_mem_sh(PyObject *self, PyObject *args) 
{
	char * string_shm;
	if (!PyArg_ParseTuple(args, "s", &string_shm)) {
		PyErr_SetString(PyExc_RuntimeError, "set_mem_sh: parse except");
	}
	bool error_open_file_flag = false;

#if defined(WIN)
	HANDLE hMapFile = OpenFileMapping(
		FILE_MAP_ALL_ACCESS,
		FALSE,
		string_shm);
	if (hMapFile == NULL) error_open_file_flag = true;
#elif defined(LINUX)
	int hMapFile = shm_open(string_shm, O_RDWR, S_IRWXU | S_IRWXG | S_IRWXO);
	if (hMapFile == -1) error_open_file_flag = true;
#endif	
	if (error_open_file_flag) {
		Py_INCREF(Py_False);
		return Py_False;
	}
	Py_INCREF(Py_True);
	return Py_True;

}

static PyObject *
create_mem_sh(PyObject *self, PyObject *args)
{
	PyObject * pyobj_for_shrdmem = nullptr;
	char * string_shm;
	if (!PyArg_ParseTuple(args, "sO", &string_shm, &pyobj_for_shrdmem)) {
		PyErr_SetString(PyExc_RuntimeError, "set_mem_sh: parse except");
	}
	PyArrayObject * array_for_shrdmem = (PyArrayObject *) pyobj_for_shrdmem;
	array_for_shrdmem = PyArray_GETCONTIGUOUS(array_for_shrdmem);
	if (array_for_shrdmem->base != nullptr) {
		PyErr_SetString(PyExc_RuntimeError, "set_mem_sh: array is not homogeneous");
	}
	/* Аrray size calculation */ 
	char * shBuf = create_shared_memory(string_shm, ARRAY_FULL_SIZE(array_for_shrdmem));
	if (shBuf == nullptr) {
		Py_INCREF(Py_None);
		return Py_None;
	}
	/* Copy array struct from heap to shared memory */
	*((size_t *) shBuf) = ARRAY_FULL_SIZE(array_for_shrdmem);
	shBuf += sizeof(size_t);
	copy_from_numpy_array_to_buffer(array_for_shrdmem, shBuf);
	Py_INCREF(Py_True);
	return Py_True;
}

static PyObject *
attach_mem_sh(PyObject *self, PyObject *args)
{
	char * string_shm;
	if (!PyArg_ParseTuple(args, "s", &string_shm)) {
		PyErr_SetString(PyExc_RuntimeError, "get_mem_sh: parse except");
	}
	char * shBuf = attach_shared_memory(string_shm);
	if (shBuf == nullptr) {
		Py_INCREF(Py_None);
		return Py_None;
	}

	PyArrayObject * array_for_shrdmem = (PyArrayObject *) shBuf;
	array_for_shrdmem = copy_from_buffer_to_numpy_array(shBuf);
	Py_INCREF((PyObject *) array_for_shrdmem);
	return (PyObject *) array_for_shrdmem;
}

static PyObject *
delete_mem_sh(PyObject *self, PyObject *args) {
	char * string_shm;
	if (!PyArg_ParseTuple(args, "s", &string_shm)) {
		PyErr_SetString(PyExc_RuntimeError, "get_mem_sh: parse except");
	}
	if (delete_shared_memory(string_shm)) {
		Py_INCREF(Py_True);
		return Py_True;
	}
	Py_INCREF(Py_False);
	return Py_False;
}

void mutex_destructor(PyObject * m_obj) {
#if defined(WIN)
	const char * name = PyCapsule_GetName(m_obj);
	HANDLE mut = (HANDLE) PyCapsule_GetPointer(m_obj, PyCapsule_GetName(m_obj));
	delete name;
#elif defined(LINUX)
	const char * name = PyCapsule_GetName(m_obj);
	sem_wrapper * mut = (sem_wrapper *) PyCapsule_GetPointer(m_obj, name);
	if (name != NULL) {
		if (mut->is_locked) {
			sem_post(mut->sem);
			mut->is_locked = false;
		}
		if (name != NULL) {
			delete name;
		}
		delete mut;
	}
#endif
}

static PyObject *
create_mutex(PyObject *self, PyObject *args) {
	bool error_open_file_flag = false;
	char * string_smp;
	if (!PyArg_ParseTuple(args, "s", &string_smp)) {
		PyErr_SetString(PyExc_RuntimeError, "create_mutex: parse except");
		return nullptr;
	}
	char * string_shm_new = new char[strlen(string_smp) + 1];
	strcpy(string_shm_new, string_smp);
#if defined(WIN)
	SECURITY_ATTRIBUTES sa;
	if (!CreateMyDACL(&sa))
    {
         // Error encountered; generate message and exit.
		PyErr_SetString(PyExc_RuntimeError, "create_mutex: failed CreateMyDACL");
		return nullptr;
    }
	HANDLE mut = CreateMutex(
		&sa, 
		FALSE, 
		string_shm_new
	);
	if (mut == nullptr) {
		error_open_file_flag = true;
	}
#elif defined(LINUX)
	sem_wrapper * mut = new sem_wrapper{
		sem_open(string_shm_new, O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO, 1), 
		false
	};
	if (mut->sem == SEM_FAILED) {
		error_open_file_flag = true;
	}
#endif
	if (error_open_file_flag) {
		Py_INCREF(Py_None);
		return Py_None;
	}
	return PyCapsule_New((void *) mut, string_shm_new, (PyCapsule_Destructor) mutex_destructor);
}

static PyObject *
open_mutex(PyObject *self, PyObject *args) {
	bool error_open_file_flag = false;
	char * string_smp;
	if (!PyArg_ParseTuple(args, "s", &string_smp)) {
		PyErr_SetString(PyExc_RuntimeError, "open_mutex: parse except");
		return nullptr;
	}
	char * string_shm_new = new char[strlen(string_smp) + 1];
	strcpy(string_shm_new, string_smp);
#if defined(WIN)
	HANDLE mut = OpenMutex(
		MUTEX_ALL_ACCESS, 
		TRUE, 
		string_shm_new
	);
	if (mut == nullptr) error_open_file_flag = true;
#elif defined(LINUX)
	sem_wrapper * mut = new sem_wrapper{
		sem_open(string_shm_new, 0), 
		false
	};
	if (mut->sem == SEM_FAILED) error_open_file_flag = true;
#endif

	if (error_open_file_flag) {
		Py_INCREF(Py_None);
		return Py_None;
	}
	return PyCapsule_New((void *) mut, string_shm_new, (PyCapsule_Destructor) mutex_destructor);
}

static PyObject *
release_mutex(PyObject *self, PyObject *args) {
	PyObject * caps_mutex;
	if (!PyArg_ParseTuple(args, "O", &caps_mutex)) {
		PyErr_SetString(PyExc_RuntimeError, "release_mutex: parse except");
		return nullptr;
	}
#if defined(WIN)
	HANDLE mut = (HANDLE) PyCapsule_GetPointer(caps_mutex, PyCapsule_GetName(caps_mutex));
	ReleaseMutex(mut);
#elif defined(LINUX)
	sem_wrapper * mut = (sem_wrapper *) PyCapsule_GetPointer(caps_mutex, PyCapsule_GetName(caps_mutex));
	if (mut->is_locked) {
		sem_post(mut->sem);
		mut->is_locked = false;
	}
#endif
	Py_INCREF(Py_True);
	return Py_True;
}

static PyObject *
close_mutex(PyObject *self, PyObject *args) {
	PyObject * caps_mutex;
	if (!PyArg_ParseTuple(args, "O", &caps_mutex)) {
		PyErr_SetString(PyExc_RuntimeError, "close_mutex: parse except");
		return nullptr;
	}
	if (caps_mutex != Py_None) {
		mutex_destructor(caps_mutex);
		Py_INCREF(Py_True);
		return Py_True;
	} else {
		Py_INCREF(Py_False);
		return Py_False;	
	}
}

static PyObject *
remove_mutex(PyObject *self, PyObject *args) {
	PyObject * caps_mutex;
	if (!PyArg_ParseTuple(args, "O", &caps_mutex)) {
		PyErr_SetString(PyExc_RuntimeError, "close_mutex: parse except");
		return nullptr;
	}
#if defined(WIN)
	Py_INCREF(Py_True);
	return Py_True;
#elif defined(LINUX)
	sem_wrapper * mut = (sem_wrapper *) PyCapsule_GetPointer(caps_mutex, PyCapsule_GetName(caps_mutex));
	const char * name = PyCapsule_GetName(caps_mutex);
	if (sem_unlink(name) == -1) {
		Py_INCREF(Py_False);
		return Py_False;
	}
	if (name != NULL) {
		delete name;
	}
	delete mut;
	PyCapsule_SetName(caps_mutex, NULL);
	Py_INCREF(Py_True);
	return Py_True;
#endif	
}

static PyObject * _try_capture_mutex(PyObject * caps_mutex, int msec) {
#if defined(WIN)
	HANDLE mut = (HANDLE) PyCapsule_GetPointer(caps_mutex, PyCapsule_GetName(caps_mutex));
	DWORD out;
	if (msec == -1) {
		out = WaitForSingleObject(mut, INFINITE);
	} else {
		out = WaitForSingleObject(mut, (DWORD) msec);
	}
#elif defined(LINUX)
	sem_wrapper * mut = (sem_wrapper *) PyCapsule_GetPointer(caps_mutex, PyCapsule_GetName(caps_mutex));
	int out;
	if (msec == 0) {
		out = sem_trywait(mut->sem);
	} else if (msec != -1) {
		timespec ts;
		ts.tv_nsec = msec * 1000;
		out = sem_timedwait(mut->sem, &ts);
	} else {
		out = sem_wait(mut->sem);
	}
	if (out == 0) mut->is_locked = true;
#endif

	if (out == 0) {
		Py_INCREF(Py_True);
		return Py_True;
	}
	Py_INCREF(Py_False);
	return Py_False;
}

static PyObject *
try_capture_mutex(PyObject *self, PyObject *args) {
	PyObject * caps_mutex;
	int timeout;
	if (!PyArg_ParseTuple(args, "Oi", &caps_mutex, &timeout)) {
		PyErr_SetString(PyExc_RuntimeError, "try_capture_mutex: parse except");
		return nullptr;
	}
	return _try_capture_mutex(caps_mutex, timeout);
}

static PyObject *
capture_mutex(PyObject *self, PyObject *args) {
	PyObject * caps_mutex;
	if (!PyArg_ParseTuple(args, "O", &caps_mutex)) {
		PyErr_SetString(PyExc_RuntimeError, "capture_mutex: parse except");
		return nullptr;
	}
	return _try_capture_mutex(caps_mutex, -1);
}

static PyObject *
get_last_error(PyObject *self, PyObject *args) {
#if defined(WIN)
	PyObject * py_err = Py_BuildValue("i", (unsigned int) GetLastError());
#elif defined(LINUX)
	PyObject * py_err = Py_BuildValue("i", (unsigned int) errno);
#endif
	Py_INCREF(py_err);
	return py_err;
}

// static PyObject *
// test_function(PyObject *self, PyObject *args) {
// 	int fd = shm_open("/test_sh_m", O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
// 	return Py_None;
// }

static PyMethodDef NumpySharedArrayMethods[] = {

    {"create_mem_sh",  create_mem_sh, METH_VARARGS,
     "method for create shared memory named."},
    {"attach_mem_sh",  attach_mem_sh, METH_VARARGS,
     "method for get shared memory named."},
    {"delete_mem_sh",  delete_mem_sh, METH_VARARGS,
     "method for del shared memory named."},
    {"check_mem_sh",  check_mem_sh, METH_VARARGS,
     "method for check shared memory named."},
    {"create_mutex",  create_mutex, METH_VARARGS,
     ""},
    {"open_mutex",  open_mutex, METH_VARARGS,
     ""},
    {"release_mutex",  release_mutex, METH_VARARGS,
     ""},
    {"close_mutex",  close_mutex, METH_VARARGS,
     ""},
    {"try_capture_mutex",  try_capture_mutex, METH_VARARGS,
      ""},
    {"close_mutex",  close_mutex, METH_VARARGS,
     ""},
    {"remove_mutex",  remove_mutex, METH_VARARGS,
     ""},

    {"capture_mutex",  capture_mutex, METH_VARARGS,
     "capture mutex"},
    {"get_last_error",  get_last_error, METH_VARARGS,
     "returns the result of the call GetLastError() function"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef wsamodule = {
    PyModuleDef_HEAD_INIT,
    "numpysharedarray", 
    NULL, 
    -1,    
    NumpySharedArrayMethods
};

PyMODINIT_FUNC
PyInit_numpysharedarray(void)
{
	import_array();
    return PyModule_Create(&wsamodule);
}
