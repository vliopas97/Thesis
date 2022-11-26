#include "heightmap.h"

using namespace GM;

__device__ typeofHeight tempArray[MAP_DIMENSIONS * MAP_DIMENSIONS];

namespace {

	__global__ void SlideBackwardDevice(typeofHeight* initarray, typeofHeight* temporary, int rows, int columns, int offset)
	{
		uint32_t x_local = blockIdx.x*blockDim.x + threadIdx.x;
		uint32_t y_local = blockIdx.y*blockDim.y + threadIdx.y;

		if (x_local >= rows || x_local >= rows - offset || y_local >= columns) return;

		temporary[(x_local + offset) * columns + y_local] = initarray[x_local * columns + y_local];
	}

	__global__ void SlideForwardDevice(typeofHeight* initarray, typeofHeight* temporary, int rows, int columns, int offset)
	{
		uint32_t x_local = blockIdx.x*blockDim.x + threadIdx.x;
		uint32_t y_local = blockIdx.y*blockDim.y + threadIdx.y;
		if (x_local >= rows || x_local < offset || y_local >= columns) return;

		temporary[(x_local - offset) * columns + y_local] = initarray[x_local * columns + y_local];
	}

	__global__ void SlideRightwardDevice(typeofHeight* initarray, typeofHeight* temporary, int rows, int columns, int offset)
	{
		uint32_t x_local = blockIdx.x*blockDim.x + threadIdx.x;
		uint32_t y_local = blockIdx.y*blockDim.y + threadIdx.y;

		if (x_local >= rows || y_local >= columns - offset || y_local >= columns) return;

		temporary[x_local * columns + (y_local + offset)] = initarray[x_local * columns + y_local];
	}

	__global__ void SlideLeftwardDevice(typeofHeight* initarray, typeofHeight* temporary, int rows, int columns, int offset)
	{
		uint32_t x_local = blockIdx.x*blockDim.x + threadIdx.x;
		uint32_t y_local = blockIdx.y*blockDim.y + threadIdx.y;

		if (x_local >= rows || y_local < offset || y_local >= columns) return;

		temporary[x_local * columns + (y_local - offset)] = initarray[x_local * columns + y_local];
	}

	__global__ void copyFromTemp(typeofHeight* dest, typeofHeight* src, int columns)
	{
		dest[blockIdx.x * columns + threadIdx.x] = src[blockIdx.x * columns + threadIdx.x];
	}

}

__host__ __device__ void HeightMapClass::Slide(SLIDE slide)
{
#ifdef __CUDA_ARCH__
	Initialize << <dimGrid, dimBlock >> > (&tempArray[0], initValue, rows * columns, columns);
	cudaDeviceSynchronize();
	switch (slide)
	{
	case SLIDE::FWD:
		SlideBackwardDevice << <dimGrid, dimBlock >> > (matrix, &tempArray[0], rows, columns, offset);
		break;
	case SLIDE::BWD:
		SlideForwardDevice << <dimGrid, dimBlock >> > (matrix, &tempArray[0], rows, columns, offset);
		break;
	case SLIDE::RGT:
		SlideLeftwardDevice << <dimGrid, dimBlock >> > (matrix, &tempArray[0], rows, columns, offset);
		break;
	case SLIDE::LFT:
		SlideRightwardDevice << <dimGrid, dimBlock >> > (matrix, &tempArray[0], rows, columns, offset);
		break;
	}
	cudaDeviceSynchronize();
	copyFromTemp << <rows, columns >> > (matrix, &tempArray[0], columns);
#else

	if (GPUOpt == GPUOPTIMIZATION::ENABLED)
	{
		typeofHeight* d_matrix, *temp;
		cudaMalloc((void**)&d_matrix, rows * columns * sizeof(typeofHeight));
		cudaMalloc((void**)&temp, rows * columns * sizeof(typeofHeight));
		cudaMemcpy(d_matrix, matrix, rows * columns * sizeof(typeofHeight), cudaMemcpyHostToDevice);
		Initialize << < dimGrid, dimBlock >> > (temp, initValue, rows * columns, columns);

		switch (slide)
		{
		case SLIDE::FWD:
			SlideBackwardDevice << <dimGrid, dimBlock >> > (d_matrix, temp, rows, columns, offset);
			break;
		case SLIDE::BWD:
			SlideForwardDevice << <dimGrid, dimBlock >> > (d_matrix, temp, rows, columns, offset);
			break;
		case SLIDE::RGT:
			SlideLeftwardDevice << <dimGrid, dimBlock >> > (d_matrix, temp, rows, columns, offset);
			break;
		case SLIDE::LFT:
			SlideRightwardDevice << <dimGrid, dimBlock >> > (d_matrix, temp, rows, columns, offset);
			break;
		}
		cudaDeviceSynchronize();
		cudaMemcpy(matrix, temp, rows * columns * sizeof(typeofHeight), cudaMemcpyDeviceToHost);
		cudaFree(d_matrix);
		cudaFree(temp);
	}
	else
	{
		switch (slide)
		{
		case SLIDE::FWD:
			SlideBackwardHost();
			break;
		case SLIDE::BWD:
			SlideForwardHost();
			break;
		case SLIDE::RGT:
			SlideLeftwardHost();
			break;
		case SLIDE::LFT:
			SlideRightwardHost();
			break;
		}
	}
#endif

}

__host__ void GM::HeightMapClass::SlideBackwardHost()
{
	for (int j = 0; j < columns; j++)
	{
		for (int i = rows - offset - 1; i >= 0; i--)
		{
			matrix[(i + offset) * columns + j] = matrix[i * columns + j];
		}
		for (int i = 0; i < offset; i++)
		{
			matrix[i * columns + j] = initValue;
		}
	}
}


__host__ void GM::HeightMapClass::SlideForwardHost()
{
	for (int j = 0; j < columns; j++)
	{
		for (int i = offset; i < rows; i++)
		{
			matrix[(i - offset) * columns + j] = matrix[i * columns + j];
		}
		for (int i = rows - offset; i < rows; i++)
		{
			matrix[i * columns + j] = initValue;
		}
	}
}

__host__ void GM::HeightMapClass::SlideRightwardHost()
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = columns - offset - 1; j >= 0; j--)
		{
			matrix[i * columns + (j + offset)] = matrix[i * columns + j];
		}
		for (int j = 0; j < offset; j++)
		{
			matrix[i * columns + j] = initValue;
		}
	}
}

__host__ void GM::HeightMapClass::SlideLeftwardHost()
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = offset; j < columns; j++)
		{
			matrix[i * columns + (j - offset)] = matrix[i * columns + j];
		}
		for (int j = columns - offset; j < columns; j++)
		{
			matrix[i * columns + j] = initValue;
		}
	}
}
