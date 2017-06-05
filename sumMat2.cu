#include <iostream>
#include <stdio.h>

double ** crearMat(int m, int n)
{
    int i;
    double **A;
    A = (double **) malloc((size_t)(m*sizeof(double *)));
    A[0] = (double *) malloc((size_t)((m*n)*sizeof(double)));
    for(i=1; i<=m; i++){
        A[i]=A[i-1] + n;
    }
    return A;
}

void imprimirMat(double **A, int m, int n)
{
    int i, j;
    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            std::cout << A[i][j] << " ";
        }
        std::cout << "\n";
    }
}

int validarMat(double **A, int m, int n)
{
    int i, j;
    for(i=0; i<m; i++)
        for(j=0; j<n; j++)
            if (A[i][j] != (5+9)) {printf("error en %d, %d, valor: %f\n", i,j,A[i][j]); return 0;}
    return 1;
}

__global__ void sumMat(double *A, double *B, double *C, int N)
{
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;

    if( (col < N) && (row < N)){
        C[col*N + row] = A[col*N + row] + B[col*N + row];
        //C[col][row] = B[col][row] + A[col][row];
    }

}



int main()
{
    const int N = 20;
    double **h_A,**h_B, **h_C;
    h_A = crearMat(N,N);
    h_B = crearMat(N,N);
    h_C = crearMat(N,N);
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            h_A[i][j]=5;
            h_B[i][j]=9;
            h_C[i][j]=0;
        }
    }

    double *d_A,*d_B,*d_C;

    cudaMalloc(&d_A, N*N*sizeof(double));
    cudaMalloc(&d_B, N*N*sizeof(double));
    cudaMalloc(&d_C, N*N*sizeof(double));
    cudaMemcpy(d_A, h_A[0], N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B[0], N*N*sizeof(double), cudaMemcpyHostToDevice);

    dim3 BlockSize(16,16);
    dim3 GridSize((N+BlockSize.x-1)/BlockSize.x,(N+BlockSize.y-1)/BlockSize.y);

    sumMat<<<GridSize, BlockSize>>>(d_A,d_B,d_C,N);

    cudaMemcpy(h_C[0], d_C,N*N*sizeof(double),cudaMemcpyDeviceToHost);
    imprimirMat(h_C,N,N);
    if (!validarMat(h_C, N, N)) printf("Error!\n");
    else printf("Todo Bien!\n");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
