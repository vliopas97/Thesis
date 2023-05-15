#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "BasicTypes.h"
#include "sl/Camera.hpp"

namespace GM {

	/**
	\brief Initializes the array on the device
	 */
	template<typename T>
	__global__ void Initialize(T* matrix, T initvalue, unsigned int rows, unsigned int columns)
	{
		uint32_t x_local = blockIdx.x*blockDim.x + threadIdx.x;
		uint32_t y_local = blockIdx.y*blockDim.y + threadIdx.y;

		if (x_local >= rows || y_local >= columns) return;

		matrix[x_local * columns + y_local] = initvalue;
	}

	/**
	\class MatClassBase
	\brief Represents the base class for MatClass
	 */
	class MatClassBase
	{
	public:
		enum class GPUOPTIMIZATION
		{
			DEFERRED,
			DISABLED,
			ENABLED
		};
	};

	/**
	\class MatClass
	\brief Represents a two dimensions array for both CPU and GPU.
	
	It is defined in an row-major order but for the accessing of elements MatClass perceives the rows in inverse order, 
	it means that, the entire first row stored in the buffer is seen as the last, 
	followed by the entire second row seen as second last, and so on.
	 *
	 * | | | | |
	 * |-|-|-|-|
	 * | an0 | an1 | ... | anm |
	 * | ... | ... | ... | ... |
	 * | a10 | a21 | ... | a1m |
	 * | a00 | a01 | ... | a0m |
	 *
	 */
	template<typename T>
	class MatClass : public MatClassBase
	{
	public:
		
		/**
		\brief MatClass constructor
		@param rows: the rows of the matrix
		@param columns: the columns of the matrix
		@param initValue: value needed for the matrix initialization
		@param GPUOpt: GPU Optimization policy for matrix processing (automatically enabled if matrix created GPU)
		 */
		__host__ __device__ MatClass(unsigned int rows, unsigned int columns, T initValue, GPUOPTIMIZATION GPUOptIn = GPUOPTIMIZATION::DEFERRED);

		/**
		\brief MatClass constructor
		@param rows: the rows of the matrix
		@param columns: the columns of the matrix
		@param matrix: the C-style matrix to be used
		 */
		__host__ __device__ MatClass(unsigned int rows, unsigned int columns, T* matrix);

		__host__ __device__ virtual ~MatClass();

		/**
		\brief Provides access to a specific point in the matrix.
		\param i : specify the row
		\param j : specify the column
		 */
		__host__ __device__ T& operator() (unsigned int i, unsigned int j);

		/**
		\brief Returns the value of a specific point in the matrix.
		\param i : specify the column
		\param j : specify the row
		 */
		__host__ __device__ T operator() (unsigned int i, unsigned int j) const;

	private:

		__host__ __device__ void init();

	public:

		unsigned int rows, columns;

		GPUOPTIMIZATION GPUOpt;

		dim3 dimGrid, dimBlock;

		T initValue;

		T* matrix;
	};

}

extern template class GM::MatClass<int>;
extern template class GM::MatClass<float>;
extern template class GM::MatClass<GM::FLOOR>;