#pragma once

#include <mutex>
#include <memory>
#include "heightmap.h"

namespace GM
{
	/**
	\class GlobalMapDevice
	\brief Global Map data singleton class stored in GPU for performance reasons
	 */
	class GlobalMapDevice
	{
	public:

		__host__ __device__ GlobalMapDevice(const GlobalMapDevice&) = delete;

		__host__ __device__ GlobalMapDevice& operator=(const GlobalMapDevice&) = delete;

		/**
		\brief Host function that returns a pointer of the aray on the device
		@return The arrays pointer in the GPU
		 */
		static __host__ GlobalMapDevice** get();

		/**
		\brief Decides if the GlobalMap will shift based on the movement of the camera/user
		@param camerapos : The camera/user position
		 */
		__device__ void checkMovementInternal(sl::float3 camerapos);

		/**
		\brief Updates the array's limits on the event of a shift
		@param slide : The shifting direction
		 */
		__device__ void updateInternal(const SLIDE& slide);

		/**
		\brief Returns the value of a specific point in the matrix
		@param i : Specify the row
		@param j : Specify the column
		 */
		__device__ typeofHeight& operator() (unsigned int i, unsigned int j);

		/**
		\brief Copies a part of Global Map to CPU
		@param location : The entry point specifies the part of the Global Map to be copied, in world coords
		@param entrance : The direction in world coords from the entry point helps to specify the part of Global Map to be copied
		@param i_low : [out] the first row of the matrix to be copied
		@param j_low : [out] the first column of the matrix to be copied
		@return : a pointer to a C-style array holding the copied part of the Global Map
		*/
		static __host__ typeofHeight* copyGlobalMap(sl::float2 location, ENTRANCE_DIR entrance, int& i_low, int& j_low);

		/**
		\brief Copies a part of Global Map to CPU for initializing the game and does not need player's direction
		@param location : The entry point specifies the part of the Global Map to be copied, in world coords
		@param i_low : [out] the first row of the matrix to be copied
		@param j_low : [out] the first column of the matrix to be copied
		@return : a pointer to a C - style array holding the copied part of the Global Map
		*/
		static __host__ typeofHeight* copyGlobalMap(sl::float2 location, int& i_low, int& j_low);

		/**
		\brief Updates the GlobalMap based on the numerical data from the newly grabbed camera frame
		@param pose : The pose of the camera in current frame
		@param mat : The matrix holding the 3D point cloud for the current camera frame
		@param planeEquation : The floor plane found for the current camera frame
		 */
		static __host__ cudaError_t editGlobalMap(sl::Mat& mat, const sl::float4& planeEquation, const sl::Transform pose = sl::Transform());

		/**
		\brief Holds a static mutex in host for providing thread safety from the side of the CPU
		\Note This mutex must be locked before every read/write operation on the GlobalMap if it is accessed through multiple threads
		*/
		static __host__ std::mutex& mutex();

		/**
		\brief Provides access to the current array boundary, stored in the GPU, to the host
		@return boundary : The boundary stored in the CPU side
		*/
		static __host__ boundary boundaryToHost();

		/**
		\brief Scan area in physical space that corresponds to the virtual room for any dynamic obstacles arising
		Updates the information regarding the room's areas and if their status changes to/from obstacle
		@param vec : Array containing each piece of area compirising the room in physical space coordinates
		*/
		static __host__ void scanForObstacles(std::vector<GlobalMappingInformation>& vec);

		/**
		\brief Takes a 3-d point cloud given in reference to the camera position and converts it to world coordinates
		@param pointcloud : The pointcloud given in sl::REFERENCE_FRAME::CAMERA
		@param transform : The world transform corresponding to the camera's position and orientation in space
		*/
		static __host__ cudaError_t transform(sl::Mat& pointcloud,const sl::Transform& transform);

		/**
		\brief Takes the floor plane for a 3-d point cloud given in reference to the camera position 
		and converts it to world coordinates
		@param plane : The plane given in sl::REFERENCE_FRAME::CAMERA
		@param transform : The world transform corresponding to the camera's position and orientation in space
		@return planeEquation : The equation of the plane expressed in the World Reference System
		*/
		static __host__ sl::float4 transform(const sl::float4& planeEquation, const sl::Transform& transform);

	private:

		__device__ GlobalMapDevice();


	public:

		HeightMapClass heightmap;

		limit lim;

		boundary boundary;
	private:

		friend __global__ void createGlobalMap(GlobalMapDevice** ptr);

	};

}
//__global__ void insertPointCloudInternal(GM::GlobalMapDevice ** ptr, sl::float4* mat, unsigned int step, unsigned int width, unsigned height, sl::float4* planeEquation);