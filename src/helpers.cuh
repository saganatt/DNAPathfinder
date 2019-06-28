#ifndef __HELPERS_H__
#define __HELPERS_H__

// Atomic max for float
// Computes max of *address and val and stores the result at address
// Robert Crovella's code from: https://stackoverflow.com/questions/17371275/implementing-max-reduce-in-cuda
__device__
float atomicMaxf(float* address, float val) {
    int32_t *address_as_int =(int32_t*)address;
    int32_t old = *address_as_int, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
    }
    return __int_as_float(old);
}

// Atomic min for float
// Computes min of *address and val and stores the result at address
__device__
float atomicMinf(float* address, float val) {
    int32_t *address_as_int =(int32_t*)address;
    int32_t old = *address_as_int, assumed;
    while (val < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
    }
    return __int_as_float(old);
}

// += overload for volatile float3
inline
__device__
void operator+=(volatile float3 &a, volatile float3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

#endif // __HELPERS_H__
