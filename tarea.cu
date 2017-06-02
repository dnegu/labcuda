/*David Neyra Gutierrez*/
#include <stdio.h>
#include <cutil.h>

#define MAX_DATA_SIZE		1024*1024*32		

void GoldenBrick(float *pA, float *pB, float *pResult, int count)
{
	for (int i=0; i < count; i++)
	{
		pResult[count] = sqrt(pA[count] * pB[count] / 12.34567) * sin(pA[count]);
	}
}

__global__ void multiplyNumbersGPU(float *pDataA, float *pDataB, float *pResult)
{
	int tid = (blockIdx.y * 128 * 256) + blockIdx.x * 256 + threadIdx.x;	
	pResult[tid] = sqrt(pDataA[tid] * pDataB[tid] / 12.34567) * sin(pDataA[tid]);

}

int main(int argc, char **argv){
	float *h_dataA, *h_dataB, *h_resultC;
	float *d_dataA, *d_dataB, *d_resultC;
	double gpuTime;
    int i;

    unsigned int hTimer;

    CUT_DEVICE_INIT(argc, argv);
    CUT_SAFE_CALL(cutCreateTimer(&hTimer));

    printf("Inicializando Datos...\n");
	h_dataA     = (float *)malloc(sizeof(float) * MAX_DATA_SIZE);
	h_dataB     = (float *)malloc(sizeof(float) * MAX_DATA_SIZE);
	h_resultC = (float *)malloc(sizeof(float) * MAX_DATA_SIZE);
	CUDA_SAFE_CALL( cudaMalloc( (void **)&d_dataA, sizeof(float) * MAX_DATA_SIZE) );
	CUDA_SAFE_CALL( cudaMalloc( (void **)&d_dataB, sizeof(float) * MAX_DATA_SIZE) );
	CUDA_SAFE_CALL( cudaMalloc( (void **)&d_resultC , sizeof(float) * MAX_DATA_SIZE) );

	srand(123);
	for(i = 0; i < MAX_DATA_SIZE; i++)
	{
		h_dataA[i] = (float)rand() / (float)RAND_MAX;
		h_dataB[i] = (float)rand() / (float)RAND_MAX;
	}

	int firstRun = 1;	
	const int useGPU = 1;	

	for (int dataAmount = MAX_DATA_SIZE; dataAmount > 128*256; dataAmount /= 2)
	{
		int blockGridWidth = 128;
		int blockGridHeight = (dataAmount / 256) / blockGridWidth;

		dim3 blockGridRows(blockGridWidth, blockGridHeight);
		dim3 threadBlockRows(256, 1);

        CUT_SAFE_CALL( cutResetTimer(hTimer) );
        CUT_SAFE_CALL( cutStartTimer(hTimer) );

		if (useGPU == 1)
		{

			// copiando datos a Device
			CUDA_SAFE_CALL( cudaMemcpy(d_dataA, h_dataA, sizeof(float) * dataAmount, cudaMemcpyHostToDevice) );
			CUDA_SAFE_CALL( cudaMemcpy(d_dataB, h_dataB, sizeof(float) * dataAmount, cudaMemcpyHostToDevice) );

			// Multiplicacion
			multiplyNumbersGPU<<<blockGridRows, threadBlockRows>>>(d_dataA, d_dataB, d_resultC);
			CUT_CHECK_ERROR("multiplyNumbersGPU() execution failed\n");
			CUDA_SAFE_CALL( cudaThreadSynchronize() );

			// copiando datos al host
			CUDA_SAFE_CALL( cudaMemcpy(h_resultC, d_resultC, sizeof(float) * dataAmount, cudaMemcpyDeviceToHost) );
		}
		else
		{
			GoldenBrick(h_dataA, h_dataB, h_resultC, dataAmount);
		}

		CUT_SAFE_CALL(cutStopTimer(hTimer));
		gpuTime = cutGetTimerValue(hTimer);
		if (!firstRun || !useGPU)
		{
			printf("Elementos: %d - tiempo convolucion : %f msec - %f Multiplicaciones/sec\n", dataAmount, gpuTime, blockGridHeight * 128 * 256 / (gpuTime * 0.001));
		}
		else
		{
			firstRun = 0;
			dataAmount *= 2;	
		}
	}

    printf("Limpiando memoria...\n");
	CUDA_SAFE_CALL( cudaFree(d_resultC ) );
	CUDA_SAFE_CALL( cudaFree(d_dataB) );
	CUDA_SAFE_CALL( cudaFree(d_dataA) );
	free(h_resultC);
	free(h_dataB);
	free(h_dataA);

    CUT_SAFE_CALL(cutDeleteTimer(hTimer));
    CUT_EXIT(argc, argv);
}
