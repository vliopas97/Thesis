#include "MatClass.h"

using namespace GM;

#ifdef __CUDA_ARCH__
#define GPUOPT(x) x = GPUOPTIMIZATION::ENABLED;
#else
#define GPUOPT(x) x = ( x == GPUOPTIMIZATION::DEFERRED && rows * columns > 100000) ? GPUOPTIMIZATION::ENABLED : GPUOPTIMIZATION::DISABLED;
#endif

template class MatClass<int>;
template class MatClass<float>;
template class MatClass<FLOOR>;

template<typename T>
GM::MatClass<T>::MatClass(unsigned int rows, unsigned int columns, T * matrix)
	:rows(rows), columns(columns), matrix(matrix), GPUOpt(GPUOPTIMIZATION::DISABLED)
{
}

template<typename T>
__host__ __device__ MatClass<T>::MatClass(unsigned int rows, unsigned int columns, T initValue, GPUOPTIMIZATION GPUOptIn)
	:rows(rows), columns(columns), initValue(initValue), GPUOpt(GPUOptIn)
{
	init();

	GPUOPT(GPUOpt);

#ifdef __CUDA_ARCH__
	Initialize << <dimGrid, dimBlock >> > (matrix, initValue, rows, columns);
#else
	if (GPUOpt == GPUOPTIMIZATION::DISABLED)
	{
		for (unsigned int i = 0; i < rows; ++i)
			for (unsigned int j = 0; j < columns; ++j)
				matrix[i * columns + j] = initValue;
	}
	else
	{
		T* device_matrix;
		cudaMalloc((void**)&device_matrix, rows * columns * sizeof(T));

		Initialize << <dimGrid, dimBlock >> > (device_matrix, initValue, rows, columns);
		cudaMemcpy(matrix, device_matrix, rows * columns * sizeof(T), cudaMemcpyDeviceToHost);
		cudaFree(device_matrix);
	}
#endif

}

template<typename T>
__host__ __device__  MatClass<T>::~MatClass()
{
	delete[] matrix;
}

template<typename T>
__host__ __device__ T & GM::MatClass<T>::operator()(unsigned int i, unsigned int j)
{
	return matrix[(rows - i - 1) * columns + j];
}

template<typename T>
__host__ __device__ T GM::MatClass<T>::operator()(unsigned int i, unsigned int j) const
{
	return matrix[(rows - i - 1) * columns + j];
}

template<typename T>
__host__ __device__ void GM::MatClass<T>::init()
{
	dimBlock.x = 32;
	dimBlock.y = 8;

	dimGrid.x = (rows + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (columns + dimBlock.y - 1) / dimBlock.y;

	matrix = new T[rows * columns];
}
