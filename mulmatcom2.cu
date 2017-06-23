#include <iostream>
using namespace std;

#define BLOQUE 16

typedef struct {    
	int ancho;    
	int alto;    
	int paso;    
	float* elementos; 
} Matrix;

__device__ float Getelemento (const Matrix A, int row, int col){    
return A.elementos[row * A.paso + col];
}

__device__ void Setelemento(Matrix A, int row, int col,float value){  
A.elementos[row * A.paso + col] = value;
}

/* Obtener la submatriz Asub que se encuentra col submatrices    
a la derecha y row submatrices abajo del comienzo de A*/
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {    
	Matrix Asub;    
	Asub.ancho   = BLOQUE;    
	Asub.alto   = BLOQUE;    
	Asub.paso   = A.paso;    
	Asub.elementos = &A.elementos[A.paso * BLOQUE * row+ BLOQUE * col];    
	return Asub; 
}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {    
	int blockRow = blockIdx.y;    int blockCol = blockIdx.x;    
	Matrix subMat = GetSubMatrix(C, blockRow, blockCol);   
	float temp = 0; // Variable para guardar resultado    
	int row = threadIdx.y;  int col = threadIdx.x;    

	//Bucle para multiplicar submatrices  Asubi y Bsubi    
	for (int m = 0; m < (A.ancho / BLOQUE); ++m) {            
		Matrix Asub = GetSubMatrix(A, blockRow, m);  // Obten Asub de A      
		Matrix Bsub = GetSubMatrix(B, m, blockCol);    // Obten Bsub de B   
   
	// Declara y carga variables en memoria compartida      
		__shared__ float As[BLOQUE][BLOQUE];      
		__shared__ float Bs[BLOQUE][BLOQUE];                                 
		As[row][col] = Getelemento(Asub, row, col);                                 
		Bs[row][col] = Getelemento(Bsub, row, col);
		
      __syncthreads(); // Sincroniza para asegurar carga      
	  // Multiplica Asubi y Bsubi para actualizar temp      
	  for (int e = 0; e < BLOQUE; ++e)                            
		temp += As[row][e] * Bs[e][col];      
	__syncthreads(); // Sincroniza para asegurar fin cómputo previo 
	}
  	Setelemento(subMat, row, col, temp);   // Escribe subMat a memoria global 
}



void MatMul(const Matrix A, const Matrix B, Matrix C) { 
	//Carga A a memoria global device 
	cout<<"Cargando Matriz A"<<endl;
	Matrix d_A; d_A.ancho = d_A.paso = A.ancho; d_A.alto = A.alto; 
	size_t size = A.ancho * A.alto * sizeof(float); 
	cudaMalloc((void**)&d_A.elementos, size); 
	cudaMemcpy(d_A.elementos, A.elementos, size, cudaMemcpyHostToDevice); 
	//Carga B a memoria global device      
	cout<<"Cargando Matriz B"<<endl;
 	Matrix d_B; d_B.ancho = d_B.paso = B.ancho; d_B.alto = B.alto; 
	size = B.ancho * B.alto * sizeof(float);
	cudaMalloc((void**)&d_B.elementos, size); 
	cudaMemcpy(d_B.elementos, B.elementos, size, cudaMemcpyHostToDevice); 
 	// Reserva memoria para C en device  
	Matrix d_C;d_C.ancho = d_C.paso = C.ancho; d_C.alto = C.alto; 
	size = C.ancho * C.alto * sizeof(float); 
	cudaMalloc((void**)&d_C.elementos, size);  

	// Llamada al  kernel  
	dim3 dimBlock(BLOQUE, BLOQUE);  
	dim3 dimGrid(B.ancho / dimBlock.x, A.alto / dimBlock.y);  
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);  
	// Lee C de device  
	cout<<"Asignando a matriz C"<<endl;
	cudaMemcpy(C.elementos, d_C.elementos, size,cudaMemcpyDeviceToHost);   
	// Libera memoria device 
	cudaFree (d_A.elementos) ;
      	cudaFree (d_B.elementos) ;
      	cudaFree (d_C.elementos) ;
}



int main()
{
	cout<<"Creando Matrices"<<endl;
	Matrix A,B; Matrix C;
	A.alto=A.ancho=A.paso=B.alto=B.ancho=B.paso=C.alto=C.ancho=C.paso=10;
	float mat1[100],mat2[100],mat3[100];
	cout<<"Llenando Matrices"<<endl;
	for(int i=0;i<100;++i)
	{
		mat1[i] = i%9;
		mat2[i] = i%7;	
	}
	A.elementos=mat1;
	B.elementos=mat2;
	C.elementos=mat3;
	cout<<"Multiplicando Matrices ..."<<endl;
	MatMul(A,B,C);
	cout<<"Visualizando Resultado"<<endl;
	for(int i=0;i<C.ancho;++i)
		for(int j=0;j<C.alto;++j)
			cout<<C.elementos[i*j+j]<<endl;
}
