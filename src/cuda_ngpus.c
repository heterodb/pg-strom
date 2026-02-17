#include <stdio.h>
#include <cuda.h>

static int check(CUresult rc, const char *what)
{
    if (rc != CUDA_SUCCESS) {
        return 0;
    }
    return 1;
}

int main(void)
{
    int n = 0;
    CUresult rc;

    rc = cuInit(0);
    if (!check(rc, "cuInit")) {
        printf("0\n");
        return 0;
    }

    rc = cuDeviceGetCount(&n);
    if (!check(rc, "cuDeviceGetCount")) {
        printf("0\n");
        return 0;
    }

    if (n < 0) n = 0;
    printf("%d\n", n);
    return 0;
}
