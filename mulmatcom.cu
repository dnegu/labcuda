#define BLOCK_SIZE 16

typedef struct {    
	int width;    
	int height;    
	int stride;    
	float* elements; 
} Matrix;


// Obtener un elemento de una matriz A 
__device__ float GetElement (const Matrix A, int row, int col){    
return A.elements[row * A.stride + col];
}

// Fijar el valor de un elemento de una matriz A
__device__ void SetElement(Matrix A, int row, int col,float value){  
A.elements[row * A.stride + col] = value;
}

/* Obtener la submatriz Asub que se encuentra col submatrices    
a la derecha y row submatrices abajo del comienzo de A*/
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {    
	Matrix Asub;    
	Asub.width    = BLOCK_SIZE;    
	Asub.height   = BLOCK_SIZE;    
	Asub.stride   = A.stride;    
	Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row+ BLOCK_SIZE * col];    
	return Asub; 
}

void MatMul(const Matrix A, const Matrix B, Matrix C) { 
//Carga A a memoria global device 
	Matrix d_A; d_A.width = d_A.stride = A.width; d_A.height = A.height; 
	size_t size = A.width * A.height * sizeof(float); 
	cudaMalloc((void**)&d_A.elements, size); 
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice); 
//Carga B a memoria global device      
 ...........  
 // Reserva memoria para C en device  
	Matrix d_C;d_C.width = d_C.stride = C.width; d_C.height = C.height; 
	size = C.width * C.height * sizeof(float); cudaMalloc((void**)&d_C.elements, size);  

// Llamada al  kernel  
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);  
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);  
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);  
// Lee C de device  
	cudaMemcpy(C.elements, d_C.elements, size,cudaMemcpyDeviceToHost);   
// Libera memoria device ...

}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {    
	int blockRow = blockIdx.y;    int blockCol = blockIdx.x;    
	Matrix Csub = GetSubMatrix(C, blockRow, blockCol);   
	float Cvalue = 0; // Variable para guardar resultado    
	int row = threadIdx.y;  int col = threadIdx.x;    
	//Bucle para multiplicar submatrices  Asubi y Bsubi    
	for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {            
		Matrix Asub = GetSubMatrix(A, blockRow, m);  // Obten Asub de A      
		Matrix Bsub = GetSubMatrix(B, m, blockCol);    // Obten Bsub de B      
	// Declara y carga variables en memoria compartida      
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];      
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];                                 
		As[row][col] = GetElement(Asub, row, col);                                 
		Bs[row][col] = GetElement(Bsub, row, col);
		
      __syncthreads(); // Sincroniza para asegurar carga      
	  // Multiplica Asubi y Bsubi para actualizar Cvalue      
	  for (int e = 0; e < BLOCK_SIZE; ++e)                            
		Cvalue += As[row][e] * Bs[e][col];      
	__syncthreads(); // Sincroniza para asegurar fin cómputo previo 
	}
  SetElement(Csub, row, col, Cvalue);   // Escribe Csub a memoria global 
}

int main()
{
	Matrix
}
