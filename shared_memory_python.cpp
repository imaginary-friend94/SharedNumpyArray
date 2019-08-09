#include "Python.h"
#include "numpy/arrayobject.h"
#include <windows.h>
#include <iostream>
#include <cstdio>

#define WIN_SMEM "WINDOWS SHARED MEMORY"
#define ARRAY_STRUCT_SIZE sizeof(PyArrayObject)
#define ARRAY_FULL_SIZE(arr) (size_data_array(arr) + sizeof(int) + arr->nd * sizeof(npy_intp) * 3 + sizeof(int))

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
	//max_buffer_size *= 2;
	HANDLE hMapFile;
	hMapFile = CreateFileMapping(
		INVALID_HANDLE_VALUE,
		NULL,
		PAGE_READWRITE,
		0,
		max_buffer_size,
		string_shm);

	if (hMapFile == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "memory not allocated");
		return nullptr;
	}

	char * pBuf = (char *) MapViewOfFile(hMapFile,
                        FILE_MAP_ALL_ACCESS,
                        0,
                        0,
                        max_buffer_size);

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
	//max_buffer_size *= 2;
	HANDLE hMapFile = OpenFileMapping(
		FILE_MAP_ALL_ACCESS,
		FALSE,
		string_shm);

	if (!CloseHandle(hMapFile)) {
		return false;
	}

	// if (!UnmapViewOfFile((LPCVOID) pBuf)) {
	// 	return false;
	// }
	return true;
}

/*
 * Attach a buffer in shared memory
 */
char * attach_shared_memory(char * string_shm) {
	//max_buffer_size *= 2;
	HANDLE hMapFile = OpenFileMapping(
		FILE_MAP_ALL_ACCESS,
		FALSE,
		string_shm); 

	if (hMapFile == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "memory not attached");
		return nullptr;
	}
	char * pBuf = (char *) MapViewOfFile(hMapFile,
                        FILE_MAP_ALL_ACCESS,
                        0,
                        0,
                        sizeof(size_t));

	size_t full_array_size = *((size_t *) pBuf);

	UnmapViewOfFile((LPCVOID) pBuf);

	pBuf = (char *) MapViewOfFile(hMapFile,
                        FILE_MAP_ALL_ACCESS,
                        0,
                        0,
                        full_array_size);

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
	HANDLE hMapFile = OpenFileMapping(
		FILE_MAP_ALL_ACCESS,
		FALSE,
		string_shm);
	if (hMapFile == nullptr) {
		return Py_False;
	}
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
	if (array_for_shrdmem->base != nullptr) {
		PyErr_SetString(PyExc_RuntimeError, "set_mem_sh: array is not homogeneous");
	}
	/* Аrray size calculation */
	std::size_t size_array_bytes = size_data_array(array_for_shrdmem);   
	char * shBuf = create_shared_memory(string_shm, ARRAY_FULL_SIZE(array_for_shrdmem));
	if (shBuf == nullptr) {
		return NULL;
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
		return NULL;
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
	Py_INCREF(Py_True);
	return Py_False;
}

void destructor_mutex(PyObject *caps_mutex) {
	HANDLE mut = (HANDLE) PyCapsule_GetPointer(caps_mutex, PyCapsule_GetName(caps_mutex));
	ReleaseMutex(mut);
	//std::cout << ReleaseMutex(mut) << std::endl;
	//std::cout << GetLastError() << std::endl;
}

static PyObject *
create_mutex(PyObject *self, PyObject *args) {
	char * string_smp;
	if (!PyArg_ParseTuple(args, "s", &string_smp)) {
		PyErr_SetString(PyExc_RuntimeError, "create_mutex: parse except");
		return nullptr;
	}
	HANDLE mut = CreateMutex(
		NULL, 
		TRUE, 
		string_smp
	);
	if (mut == nullptr) {
		Py_INCREF(Py_None);
		return Py_None;
	}
	return PyCapsule_New((void *) mut, string_smp, (PyCapsule_Destructor) destructor_mutex);
}

static PyObject *
open_mutex(PyObject *self, PyObject *args) {
	char * string_smp;
	if (!PyArg_ParseTuple(args, "s", &string_smp)) {
		PyErr_SetString(PyExc_RuntimeError, "open_mutex: parse except");
		return nullptr;
	}
	HANDLE mut = OpenMutex(
		MUTEX_ALL_ACCESS, 
		TRUE, 
		string_smp
	);
	if (mut == nullptr) {
		Py_INCREF(Py_None);
		return Py_None;
	}
	return PyCapsule_New((void *) mut, string_smp, (PyCapsule_Destructor) destructor_mutex);
}

static PyObject *
release_mutex(PyObject *self, PyObject *args) {
	PyObject * caps_mutex;
	if (!PyArg_ParseTuple(args, "O", &caps_mutex)) {
		PyErr_SetString(PyExc_RuntimeError, "release_mutex: parse except");
		return nullptr;
	}
	destructor_mutex(caps_mutex);
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
	HANDLE mut = (HANDLE) PyCapsule_GetPointer(caps_mutex, PyCapsule_GetName(caps_mutex));
	if (CloseHandle(mut)) {
		Py_INCREF(Py_True);
		return Py_True;
	}
	Py_INCREF(Py_False);
	return Py_False;
}

static PyObject * _try_capture_mutex(PyObject * caps_mutex, DWORD msec) {
	HANDLE mut = (HANDLE) PyCapsule_GetPointer(caps_mutex, PyCapsule_GetName(caps_mutex));
	DWORD out = WaitForSingleObject(mut, msec);
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
	return _try_capture_mutex(caps_mutex, (DWORD) timeout);
}

static PyObject *
capture_mutex(PyObject *self, PyObject *args) {
	PyObject * caps_mutex;
	if (!PyArg_ParseTuple(args, "O", &caps_mutex)) {
		PyErr_SetString(PyExc_RuntimeError, "capture_mutex: parse except");
		return nullptr;
	}
	return _try_capture_mutex(caps_mutex, INFINITE);
}

static PyMethodDef WinSharedArrayMethods[] = {

    {"create_mem_sh",  create_mem_sh, METH_VARARGS,
     "method for create shared memory named."},
    {"attach_mem_sh",  attach_mem_sh, METH_VARARGS,
     "method for get shared memory named."},
    {"delete_mem_sh",  delete_mem_sh, METH_VARARGS,
     "method for del shared memory named."},
    {"check_mem_sh",  check_mem_sh, METH_VARARGS,
     "method for check shared memory named."},
    {"create_mutex",  create_mutex, METH_VARARGS,
     "tt"},
    {"open_mutex",  open_mutex, METH_VARARGS,
     "tt"},
    {"release_mutex",  release_mutex, METH_VARARGS,
     "tt"},
    {"close_mutex",  close_mutex, METH_VARARGS,
     "tt"},
    {"try_capture_mutex",  try_capture_mutex, METH_VARARGS,
     "tt"},
    {"capture_mutex",  capture_mutex, METH_VARARGS,
     "tt"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef wsamodule = {
    PyModuleDef_HEAD_INIT,
    "winsharedarray", 
    NULL, 
    -1,    
    WinSharedArrayMethods
};

PyMODINIT_FUNC
PyInit_winsharedarray(void)
{
	import_array();
    return PyModule_Create(&wsamodule);
}