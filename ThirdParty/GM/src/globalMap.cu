#include "cuda_runtime.h"
#include "globalMap.h"

#define index(i, j, N)  ((i)*(N)) + (j)
#define blockSize 256

using namespace GM;

namespace
{
	__device__ void warpReduce(volatile float * sdata, unsigned int tid)
	{
		sdata[tid] = fmin(sdata[tid], sdata[tid + 32]);
		sdata[tid] = fmin(sdata[tid], sdata[tid + 16]);
		sdata[tid] = fmin(sdata[tid], sdata[tid + 8]);
		sdata[tid] = fmin(sdata[tid], sdata[tid + 4]);
		sdata[tid] = fmin(sdata[tid], sdata[tid + 2]);
		sdata[tid] = fmin(sdata[tid], sdata[tid + 1]);
	}

	__global__ void findMinOfArray(sl::float4 *matrix, int N, float *op_data, int num_blocks)
	{

		unsigned int unique_id = blockIdx.x * blockDim.x + threadIdx.x; /* unique id for each thread in the block*/
		unsigned int row = unique_id % N; /*row number in the matrix*/
		unsigned int col = unique_id / N; /*col number in the matrix*/

		unsigned int thread_id = threadIdx.x; /* thread index in the block*/

		__shared__ float minChunk[blockSize];

		if ((row >= 0) && (row < N) && (col >= 0) && (col < N))
		{
			minChunk[thread_id] = matrix[index(row, col, N)].z;
		}

		__syncthreads();

		for (unsigned int stride = (blockDim.x / 2); stride > 32; stride /= 2)
		{
			__syncthreads();

			if (thread_id < stride)
			{
				minChunk[thread_id] = fmin(minChunk[thread_id], minChunk[thread_id + stride]);
			}
		}

		if (thread_id < 32)
		{
			warpReduce(minChunk, thread_id);
		}

		if (thread_id == 0)
		{
			op_data[index(0, blockIdx.x, num_blocks)] = minChunk[0];
		}
	}

	__global__ void findMin(sl::float4* mat, int N, float* result)
	{
		unsigned int block_size = blockSize;
		unsigned int grid_size = ceil((N*N) / (block_size*1.0));

		float* shared = new float[grid_size];
		findMinOfArray << <grid_size, block_size >> > (mat, N, shared, grid_size);

		__syncthreads();
		*result = FLT_MAX;
		for (int i = 0; i < grid_size; i++)
		{
			*result = fminf(shared[i], *result);
		}
		delete shared;
	}

	__device__ float getPlaneDistance(const sl::float3& point, sl::float4* const planeEquation)
	{
		return (fabsf(planeEquation->x * point.x + planeEquation->y *point.y + planeEquation->z *  point.z + planeEquation->w) /
			sqrt(planeEquation->x * planeEquation->x + planeEquation->y * planeEquation->y + planeEquation->z * planeEquation->z));
	}

	__device__ sl::float2 findGlobalMapElement(GlobalMapDevice ** ptr, sl::float4 & pointcloud)
	{
		auto& lim = (*ptr)->lim;
		if (!isfinite(pointcloud.x) || !isfinite(pointcloud.y) || !isfinite(pointcloud.z)
			|| pointcloud.x < lim.lowX || pointcloud.x > lim.highX || pointcloud.y < lim.lowY || pointcloud.y > lim.highY)
			return sl::float2(-1, -1);

		int row = static_cast<int>(floorf((pointcloud.x - lim.lowX) / 4.)) % MAP_DIMENSIONS;
		int column = static_cast<int>(floorf((pointcloud.y - lim.lowY) / 4.)) % MAP_DIMENSIONS;

		return sl::float2(row, column);
	}
	__global__ void resetGMValuesForPointCloud(GM::GlobalMapDevice ** ptr, sl::float4* mat, unsigned int step, unsigned int width, unsigned height, sl::float4* planeEquation)
	{
		uint32_t x_local = blockIdx.x * blockDim.x + threadIdx.x;
		uint32_t y_local = blockIdx.y * blockDim.y + threadIdx.y;

		if (x_local >= width || y_local >= height) return;

		sl::float4* pointcloudvalue = &mat[x_local + y_local * step];
		sl::float2 GlobalMapCoords = findGlobalMapElement(ptr, *pointcloudvalue);

		if (GlobalMapCoords.x < 0 || GlobalMapCoords.y < 0) return;

		//DISCARD OBSTACLE HIGHER THAN TWO METERS ADD CODE
		pointcloudvalue->z = getPlaneDistance(*pointcloudvalue, planeEquation);
		if (pointcloudvalue->z > 200) return;


		//atomicExch(reinterpret_cast<int*>(&(*(*ptr))(GlobalMapCoords.x, GlobalMapCoords.y)), static_cast<int>(FLOOR::WALKABLE));
		atomicExch(reinterpret_cast<int*>(&(*(*ptr))(GlobalMapCoords.x, GlobalMapCoords.y)), 0);
	}

	__global__ void insertPointCloudInternal(GlobalMapDevice** ptr, sl::float4* mat, unsigned int step, unsigned int width, unsigned int height)
	{
		uint32_t x_local = blockIdx.x * blockDim.x + threadIdx.x;
		uint32_t y_local = blockIdx.y * blockDim.y + threadIdx.y;

		if (x_local >= width || y_local >= height) return;

		sl::float4 pointcloudvalue = mat[x_local + y_local * step];
		sl::float2 GlobalMapCoords = findGlobalMapElement(ptr, pointcloudvalue);

		if (GlobalMapCoords.x < 0 || GlobalMapCoords.y < 0) return;
		if (pointcloudvalue.z > 200) return;


		//FLOOR floor = fabsf(pointcloudvalue.z) < 20 ? FLOOR::WALKABLE : FLOOR::OBSTACLE;
		//atomicMax(reinterpret_cast<int*>(&(*(*ptr))(GlobalMapCoords.x, GlobalMapCoords.y)), static_cast<int>(floor));

		(pointcloudvalue.z >= 0) ? __int_as_float(atomicMax((int*)(&(*(*ptr))(GlobalMapCoords.x, GlobalMapCoords.y)), __float_as_int(pointcloudvalue.z))) :
			__uint_as_float(atomicMin((unsigned int*)(&(*(*ptr))(GlobalMapCoords.x, GlobalMapCoords.y)), __float_as_uint(pointcloudvalue.z)));

	}

	__global__ void RetrieveBoundaryFromDevice(GlobalMapDevice** Ptr, boundary* Boundary)
	{
		*Boundary = (*Ptr)->boundary;
	}

	__global__ void k_CopyGlobalMapInternal(GlobalMapDevice** ptr, typeofHeight* num, unsigned int rows, unsigned int columns)
	{
		int i = blockIdx.x;
		int j = threadIdx.x;
		num[(gridDim.x - i - 1) * blockDim.x + j] = (*(*ptr))(i + rows, j + columns);
	}

	__global__ void k_copyGlobalMapInternal_xDown(GlobalMapDevice** ptr, typeofHeight* num, unsigned int rows, unsigned int columns) 
	{
		int i = blockIdx.x;
		int j = threadIdx.x;
		num[i * blockDim.x + (blockDim.x - j - 1)] = (*(*ptr))(i + rows, j + columns);
	}

	__global__ void k_copyGlobalMapInternal_yUp(GlobalMapDevice** ptr, typeofHeight* num, unsigned int rows, unsigned int columns)
	{
		int i = blockIdx.x;
		int j = threadIdx.x;
		num[(gridDim.x - j - 1) * blockDim.x + (gridDim.x - i - 1)] = (*(*ptr))(i + rows, j + columns);
	}

	__global__ void k_copyGlobalMapInternal_yDown(GlobalMapDevice** ptr, typeofHeight* num, unsigned int rows, unsigned int columns)
	{
		int i = blockIdx.x;
		int j = threadIdx.x;
		num[j * blockDim.x + i] = (*(*ptr))(i + rows, j + columns);
	}

	__global__ void k_copyGlobalMap(GlobalMapDevice** ptr, typeofHeight* matrix, sl::float2 location, int* ilow, int* jlow, ENTRANCE_DIR entrance)
	{
		int row = static_cast<int>(floorf((location.x - (*(*ptr)).lim.lowX) / 4.)) % MAP_DIMENSIONS;
		int column = static_cast<int>(floorf((location.y - (*(*ptr)).lim.lowY) / 4.)) % MAP_DIMENSIONS;

		dim3 dimGrid, dimBlock;
		dimBlock.x = 32;
		dimBlock.y = 8;

		dimGrid.x = ((*(*ptr)).heightmap.rows / 4 + dimBlock.x - 1) / dimBlock.x;
		dimGrid.y = ((*(*ptr)).heightmap.columns / 4 + dimBlock.y - 1) / dimBlock.y;

		switch (entrance)
		{
		case GM::ENTRANCE_DIR::X_UP:
			*ilow = row;
			*jlow = column - 127;
			k_CopyGlobalMapInternal << <256, 256 >> > (ptr, matrix, *ilow, *jlow);
			break;
		case GM::ENTRANCE_DIR::X_DOWN:
			*ilow = row - 255;
			*jlow = column - 127;
			k_copyGlobalMapInternal_xDown << <256, 256 >> > (ptr, matrix, *ilow, *jlow);
			break;
		case GM::ENTRANCE_DIR::Y_UP:
			*ilow = row - 127;
			*jlow = column;
			k_copyGlobalMapInternal_yUp << <256, 256 >> > (ptr, matrix, *ilow, *jlow);
			break;
		case GM::ENTRANCE_DIR::Y_DOWN:
			*ilow = row - 127;
			*jlow = column - 255;
			k_copyGlobalMapInternal_yDown << <256, 256 >> > (ptr, matrix, *ilow, *jlow);
			break;
		default:
			break;
		}

	}

	__global__ void k_copyGlobalMapInit(GlobalMapDevice** ptr, typeofHeight* matrix, sl::float2 location, int* ilow, int* jlow)
	{
		int row = static_cast<int>(floorf((location.x - (*(*ptr)).lim.lowX) / 4.)) % MAP_DIMENSIONS;
		int column = static_cast<int>(floorf((location.y - (*(*ptr)).lim.lowY) / 4.)) % MAP_DIMENSIONS;

		dim3 dimGrid, dimBlock;
		dimBlock.x = 32;
		dimBlock.y = 8;

		dimGrid.x = ((*(*ptr)).heightmap.rows / 4 + dimBlock.x - 1) / dimBlock.x;
		dimGrid.y = ((*(*ptr)).heightmap.columns / 4 + dimBlock.y - 1) / dimBlock.y;

		*ilow = row - 16 /* - 127*/;
		*jlow = column - 127;

		k_CopyGlobalMapInternal << < 256, 256 >> > (ptr, matrix, *ilow, *jlow);
	}

	MatClass<FLOOR> downsample(MatClass<FLOOR>& mat)
	{
		MatClass < FLOOR > downsampled(mat.rows / downsampleFactor, mat.columns / downsampleFactor, FLOOR::DOWNSAMPLED_INIT);
		for (size_t i = 0; i < mat.rows; i++) {
			auto newrow = floor(i / downsampleFactor);
			for (size_t j = 0; j < mat.columns; j++) {
				auto newcolumn = floor(j / downsampleFactor);
				if (downsampled(newrow, newcolumn) == FLOOR::DOWNSAMPLED_INIT)
					downsampled(newrow, newcolumn) = mat(i, j);
				else if (downsampled(newrow, newcolumn) == FLOOR::UNKNOWN && mat(i, j) == FLOOR::OBSTACLE)
					downsampled(newrow, newcolumn) = FLOOR::OBSTACLE;
				else if (downsampled(newrow, newcolumn) == FLOOR::WALKABLE && mat(i, j) != FLOOR::WALKABLE) {
					downsampled(newrow, newcolumn) = mat(i, j);
				}
			}
		}
		return downsampled;
	}

	__global__ void k_preprocess(sl::float4* mat, sl::Transform* transform, unsigned int step, unsigned int width, unsigned int height)
	{
		//uint32_t x_local = blockIdx.x * blockDim.x + threadIdx.x;
		//uint32_t y_local = blockIdx.x * blockDim.y + threadIdx.y;
		uint32_t x_local = blockIdx.x;
		uint32_t y_local = threadIdx.x;
		//if (x_local == 0 && y_local == 0)
		//	printf("value %d %d \n", blockDim.x, blockDim.y);

		if (x_local >= width || y_local >= height) return;
		//printf("value is: %d, %d", x_local, y_local);

		//sl::float4* value = &mat[x_local + y_local * step];
		//*value = (*value) * (*transform);
		sl::float4 value = mat[x_local + y_local * step];
		value.w = 1;
		//if (x_local == 0 && y_local == 0)
		//{

		//}
		//printf("value %f %f %f \n", value.x, value.y, value.z);
		mat[x_local + y_local * step] = value * (*transform);
		value = mat[x_local + y_local * step];
		//printf("value %f %f %f \n", value.x, value.y, value.z);
	}

	__global__ void k_editGlobalMap(GlobalMapDevice** ptr, sl::float3 position, sl::float4* mat, unsigned int step, unsigned int width, unsigned height, sl::float4* planeEquation)
	{
		(*ptr)->checkMovementInternal(position);

		dim3 dimGrid, dimBlock;

		dimBlock.x = 32;
		dimBlock.y = 8;

		dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
		dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;

		resetGMValuesForPointCloud << <dimGrid, dimBlock >> > (ptr, mat, step, width, height, planeEquation);
		insertPointCloudInternal << <dimGrid, dimBlock >> > (ptr, mat, step, width, height);

	}

	__device__ double phi(double x)
	{
		// constants
		double a1 = 0.254829592;
		double a2 = -0.284496736;
		double a3 = 1.421413741;
		double a4 = -1.453152027;
		double a5 = 1.061405429;
		double p = 0.3275911;

		// Save the sign of x
		int sign = 1;
		if (x < 0)
			sign = -1;
		x = fabs(x) / sqrt(2.0);

		// A&S formula 7.1.26
		double t = 1.0 / (1.0 + p * x);
		double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

		return 0.5 * (1.0 + sign * y);
	}

	__global__ void k_scanForObstacles(GlobalMapDevice** ptr, GlobalMappingInformation* vec)
	{
		int I = static_cast<int>(floorf((vec[gridDim.x].position.x - (*(*ptr)).lim.lowX) / 4.)) % MAP_DIMENSIONS;
		int J = static_cast<int>(floorf((vec[gridDim.x].position.y - (*(*ptr)).lim.lowY) / 4.)) % MAP_DIMENSIONS;
		//int I = vec[gridDim.x].ilow;
		//int J = vec[gridDim.x].jlow;
		//int i = blockDim.x;
		//int j = blockDim.y;

		//auto element = static_cast<int>((*(*ptr))(I + i, J + j));
		//if (element > 0)
		//{
		//	atomicAdd(&(vec[gridDim.x].counter), 1);
		//}
		float sum = 0;
		float unm = 0;
		for (size_t i = I; i < I + 16; i++)
		{
			for (size_t j = J; j < J + 16; j++)
			{
				if ((*(*ptr))(i, j) == -1)
				{
					unm++;
				}
				else
				{
					sum += (*(*ptr))(i, j);
				}
			}
		}

		float mean = sum / (256.f - unm);
		float sum2 = 0;
		for (size_t i = I; i < I + 16; i++)
		{
			for (size_t j = J; j < J + 16; j++)
			{
				if ((*(*ptr))(i, j) != -1)
				{
					sum2 += ((*(*ptr))(i, j) - mean) * ((*(*ptr))(i, j) - mean);
				}
			}
		}

		float deviation = sqrtf(sum2 / (256.f - unm));

		if (phi((THRESHOLD - mean) / deviation) > 0.65)
		{
			vec[gridDim.x].counter = 0;
		}
		else
		{
			vec[gridDim.x].counter = 1;
		}
	}

}

__global__ void GM::createGlobalMap(GlobalMapDevice ** ptr)
{
	*ptr = new GlobalMapDevice();
}

__host__ GlobalMapDevice** GlobalMapDevice::get()
{

	static GlobalMapDevice** ptr;
	static bool flag = true;
	if (flag)
	{
		cudaMalloc((void**)&ptr, sizeof(GlobalMapDevice**));
		createGlobalMap << <1, 1 >> > (ptr);
		flag = false;
	}
	return ptr;

}

__device__ void GlobalMapDevice::checkMovementInternal(sl::float3 camerapos)
{
	//printf("boundary triggers are %f, %f, %f, %f\n", boundary.trigger.highX, boundary.trigger.lowX, boundary.trigger.highY, boundary.trigger.lowY);
	if (camerapos.x > boundary.trigger.highX)
	{
		//forward movement
		boundary.update(SLIDE::FWD);
		updateInternal(SLIDE::FWD);
	}
	else if (camerapos.x < boundary.trigger.lowX)
	{
		//backward movement
		boundary.update(SLIDE::BWD);
		updateInternal(SLIDE::BWD);
	}
	else if (camerapos.y > boundary.trigger.highY)
	{
		//rightward movement
		boundary.update(SLIDE::RGT);
		updateInternal(SLIDE::RGT);
	}
	else if (camerapos.y < boundary.trigger.lowY)
	{
		//leftward movement
		boundary.update(SLIDE::LFT);
		updateInternal(SLIDE::LFT);
	}
}

__device__ void GlobalMapDevice::updateInternal(const SLIDE & slide)
{
	switch (slide)
	{
	case SLIDE::FWD:
		lim.lowX += 2;
		lim.highX += 2;
		break;
	case SLIDE::BWD:
		lim.lowX -= 2;
		lim.highX -= 2;
		break;
	case SLIDE::RGT:
		lim.lowY += 2;
		lim.highY += 2;
		break;
	case SLIDE::LFT:
		lim.lowY -= 2;
		lim.highY -= 2;
		break;
	}
	
	heightmap.Slide(slide);
}

__device__ typeofHeight & GlobalMapDevice::operator()(unsigned int i, unsigned int j)
{
	return heightmap(i, j);
}
#define N 4

// Function to get cofactor of A[p][q] in temp[][]. n is current
// dimension of A[][]
void getCofactor(float A[N][N], float temp[N][N], int p, int q, int n)
{
	int i = 0, j = 0;

	// Looping for each element of the matrix
	for (int row = 0; row < n; row++)
	{
		for (int col = 0; col < n; col++)
		{
			//  Copying into temporary matrix only those element
			//  which are not in given row and column
			if (row != p && col != q)
			{
				temp[i][j++] = A[row][col];

				// Row is filled, so increase row index and
				// reset col index
				if (j == n - 1)
				{
					j = 0;
					i++;
				}
			}
		}
	}
}

/* Recursive function for finding determinant of matrix.
   n is current dimension of A[][]. */
float determinant(float A[N][N], int n)
{
	float D = 0; // Initialize result

	//  Base case : if matrix contains single element
	if (n == 1)
		return A[0][0];

	float temp[N][N]; // To store cofactors

	float sign = 1;  // To store sign multiplier

	 // Iterate for each element of first row
	for (int f = 0; f < n; f++)
	{
		// Getting Cofactor of A[0][f]
		getCofactor(A, temp, 0, f, n);
		D += sign * A[0][f] * determinant(temp, n - 1);

		// terms are to be added with alternate sign
		sign = -sign;
	}

	return D;
}

// Function to get adjoint of A[N][N] in adj[N][N].
void adjoint(float A[N][N], float adj[N][N])
{
	if (N == 1)
	{
		adj[0][0] = 1;
		return;
	}

	// temp is used to store cofactors of A[][]
	float sign = 1;
	float temp[N][N];

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			// Get cofactor of A[i][j]
			getCofactor(A, temp, i, j, N);

			// sign of adj[j][i] positive if sum of row
			// and column indexes is even.
			sign = ((i + j) % 2 == 0) ? 1 : -1;

			// Interchanging rows and columns to get the
			// transpose of the cofactor matrix
			adj[j][i] = (sign) * (determinant(temp, N - 1));
		}
	}
}

__host__ sl::float4 GM::GlobalMapDevice::transform(const sl::float4& planeEquation, const sl::Transform& transform)
{
	sl::Transform temp = sl::Transform(transform);
	temp.inverse();
	temp.transpose();
	return planeEquation * temp;
}

__device__ GlobalMapDevice::GlobalMapDevice()
	:heightmap(MAP_DIMENSIONS, MAP_DIMENSIONS, -1),
	lim(-2048, -2048, 2047, 2047),
	boundary(sl::float2(0, 0))
{
}

__host__ typeofHeight* GlobalMapDevice::copyGlobalMap(sl::float2 location, ENTRANCE_DIR entrance, int& i_low, int& j_low)
{
	typeofHeight* matrix;
	cudaMalloc((void**)&matrix, sizeof(float) * (MAP_DIMENSIONS / 4) * (MAP_DIMENSIONS / 4));

	int* ilow_d, *jlow_d;
	cudaMalloc((void**)&ilow_d, sizeof(int*));
	cudaMalloc((void**)&jlow_d, sizeof(int*));

	//std::unique_lock<std::mutex> lock(GlobalMapDevice::mutex());
	k_copyGlobalMap << <1, 1 >> > (GlobalMapDevice::get(), matrix, location, ilow_d, jlow_d, entrance);
	//lock.unlock();

	cudaMemcpy(&i_low, ilow_d, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&j_low, jlow_d, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(ilow_d);
	cudaFree(jlow_d);

	//array copy on host
	typeofHeight* matrix_host = new typeofHeight[(MAP_DIMENSIONS / 4) * (MAP_DIMENSIONS / 4)];
	cudaMemcpy(matrix_host, matrix, sizeof(float) * (MAP_DIMENSIONS / 4) * (MAP_DIMENSIONS / 4), cudaMemcpyDeviceToHost);
	cudaFree(matrix);

	return matrix_host;
}

__host__ typeofHeight* GM::GlobalMapDevice::copyGlobalMap(sl::float2 location, int& i_low, int& j_low)
{
	typeofHeight* matrix;
	cudaMalloc((void**)&matrix, sizeof(float) * (MAP_DIMENSIONS / 4) * (MAP_DIMENSIONS / 4));

	int* ilow_d, * jlow_d;
	cudaMalloc((void**)&ilow_d, sizeof(int*));
	cudaMalloc((void**)&jlow_d, sizeof(int*));

	//std::unique_lock<std::mutex> lock(GlobalMapDevice::mutex());
	k_copyGlobalMapInit << <1, 1 >> > (GlobalMapDevice::get(), matrix, location, ilow_d, jlow_d);
	//lock.unlock();

	cudaMemcpy(&i_low, ilow_d, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&j_low, jlow_d, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(ilow_d);
	cudaFree(jlow_d);

	//array copy on host
	typeofHeight* matrix_host = new typeofHeight[(MAP_DIMENSIONS / 4) * (MAP_DIMENSIONS / 4)];
	cudaMemcpy(matrix_host, matrix, sizeof(float) * (MAP_DIMENSIONS / 4) * (MAP_DIMENSIONS / 4), cudaMemcpyDeviceToHost);
	cudaFree(matrix);

	return matrix_host;
}

__host__ cudaError_t GlobalMapDevice::editGlobalMap(sl::Mat& mat, const sl::float4& planeEquation, const sl::Transform pose)
{
	auto ptr = GlobalMapDevice::get();
	cudaError_t cudaStatus = cudaError::cudaSuccess;
	sl::float3 camerapos = sl::float3(pose.getTranslation().x, pose.getTranslation().y, pose.getTranslation().z);
	sl::float4* planeEquationPtr;

	cudaStatus = cudaMalloc((void**)&planeEquationPtr, sizeof(sl::float4));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(planeEquationPtr);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(planeEquationPtr, &planeEquation, sizeof(sl::float4), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(planeEquationPtr);
		return cudaStatus;
	}

	////Kernel launch
	std::lock_guard<std::mutex> lock(GlobalMapDevice::mutex());
	k_editGlobalMap << <1, 1 >> > (ptr, camerapos, mat.getPtr<sl::float4>(sl::MEM::GPU), mat.getStep(sl::MEM::GPU), mat.getWidth(), mat.getHeight(), planeEquationPtr);
	
	return cudaStatus;
}

__host__ std::mutex & GlobalMapDevice::mutex()
{
	static std::mutex globalMapLock;
	return globalMapLock;
}

__host__ boundary GM::GlobalMapDevice::boundaryToHost()
{
	GM::boundary Boundary_H(sl::float2(0, 0));//on the host
	GM::boundary* Boundary_D;//on device
	cudaMalloc((void**)&Boundary_D, sizeof(boundary));

	std::unique_lock<std::mutex> lock(GlobalMapDevice::mutex());//Global Map is not Thread Safe - Need for Mutex
	RetrieveBoundaryFromDevice << <1, 1 >> > (GlobalMapDevice::get(), Boundary_D);
	lock.unlock();//release Mutex

	cudaError ErrorCode = cudaMemcpy(&Boundary_H, Boundary_D, sizeof(boundary), cudaMemcpyDeviceToHost);
	return Boundary_H;
}

__host__ void GM::GlobalMapDevice::scanForObstacles(std::vector<GlobalMappingInformation>& vec)
{
	GlobalMappingInformation* vecDevice;
	cudaMalloc((void**)&vecDevice, sizeof(GlobalMappingInformation) * vec.size());
	cudaMemcpy(vecDevice, &vec[0], sizeof(GlobalMappingInformation)*vec.size(), cudaMemcpyHostToDevice);

	//kernel launch
	dim3 grid(vec.size());
	dim3 block(16, 16);
	auto ptr = GlobalMapDevice::get();

	std::lock_guard<std::mutex> lock(GlobalMapDevice::mutex());
	k_scanForObstacles << <grid, block >> > (ptr, vecDevice);

	cudaMemcpy(&vec[0], vecDevice, sizeof(GlobalMappingInformation) * vec.size(), cudaMemcpyDeviceToHost);
}

__host__ cudaError_t GM::GlobalMapDevice::transform(sl::Mat& pointcloud,const sl::Transform& transform)
{
	cudaError_t cudaStatus = cudaError::cudaSuccess;
	sl::Transform* transformDevice;
	cudaStatus = cudaMalloc((void**)&transformDevice, sizeof(sl::Transform));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(transformDevice);
		return cudaStatus;
	}

	cudaMemcpy(transformDevice, &transform, sizeof(sl::Transform), cudaMemcpyHostToDevice);

	unsigned int step = pointcloud.getStep(sl::MEM::GPU);
	unsigned int width = pointcloud.getWidth();
	unsigned int height = pointcloud.getHeight();

	dim3 dimGrid, dimBlock;
	dimBlock.x = 32;
	dimBlock.y = 8;
	dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;

	k_preprocess <<<width, height>>> (pointcloud.getPtr<sl::float4>(sl::MEM::GPU), transformDevice, step, width, height);

	return cudaStatus;
}
