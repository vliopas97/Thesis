#pragma once
#include "sl/Camera.hpp"

#define DEBUG 0

constexpr auto MAP_DIMENSIONS = 1024;
constexpr int BOUNDARY_DIMENSIONS = 200; /*boundary box dimension for the global map update given in meters*/
constexpr int DEADBAND = 50;			/*deadband in between boundaries to prevent flickering between new and old global maps
							 actual boundary dimensions = BOUNDARY_DIMENSIONS - DEADBAND*/

constexpr int downsampleFactor = 16;//how much should the heightmap be downsampled for generation of virtual room: 16 ( 0.64m X 0.64m px) or 32 (1.28m X 1.28m px)


namespace GM
{
	constexpr float THRESHOLD = 30;

	/**
	\enum SLIDE
	\brief Lists the types of shifting that can be performed on MatClass arrays to keep the Users position in its center
	 */
	enum class SLIDE {
		FWD = 0, /**< Forward Shifting, clears bottom lines*/
		BWD, /**< Backward Shifting, clears upper lines*/
		RGT, /**< Rightward Shifting, clears leftmost columns*/
		LFT /**< Leftward Shifting, clears rightmost columns*/
	};

	/**
	\enum SLIDE
	\brief Lists the types of labeling for floor
	 */
	enum class FLOOR {
		DOWNSAMPLED_INIT = -2, /**< Init value for downsampled segment of Global Map, need to differentiate with FLOOR::UNKNOWN*/ 
		UNKNOWN = -1, /**< Unmapped area, init value for Global Map*/ 
		WALKABLE, /**< Mapped area with no obstacles*/
		OBSTACLE, /**< Mapped area with obstacles found*/
		PORTAL, /**< Mapped, walkable area bordering with unmapped area*/
		ROOM /**< Mapped, walkable area to be rendered as the actual room*/
	};

	/**
	\enum ENTRANCE_DIR
	\brief Lists the types the Doors' direction
	\Note Door's direction is the direction leading outside the room the user is currently in
	\Note Uses an extra value to differentiate between actual Portal and Doors for calculation purposes
	 */
	enum class ENTRANCE_DIR {
		WALL = -1,
		X_DOWN = 0, /**< Negative direction - X Axis */
		X_UP, /**< Positive direction - X Axis*/ 
		Y_DOWN, /**< Negative direction - Y Axis*/
		Y_UP /**< Positive direction - Y Axis*/
	};

	bool operator==(const sl::float2& a, const sl::float2& b);

	struct roomSegment;

	bool operator==(const roomSegment& a, const roomSegment& b);

	/**
	\struct limit
	\brief Groups the upper and lower limits that Global Map currently maps
	 */
	struct limit {

		__host__ __device__ limit(float lowY, float lowX, float highY, float highX);

		__host__ __device__ limit(const limit&) = default;

		__host__ __device__ limit& operator=(const limit&) = default;

		float lowY, lowX, highY, highX;

	};

	/**
	\class boundary
	\brief Rectangular block of space to track the user's movement, if its borders are crossed Global Map gets Updated
	 */
	class boundary {

	public:
		__host__ __device__ boundary(sl::float2 center = sl::float2(0, 0));

		__host__ __device__ void update(const SLIDE& slide);

		__host__ __device__ boundary(const boundary&) = default;

		__host__ __device__ boundary& operator=(const boundary & other) = default;

		sl::float2 center;

	private:
		int dim_with_deadband;
	
	public:
		limit trigger;

		friend class GlobalMapDevice;
	};

	/**
	\struct rectangleID
	\brief Useful for Calculating the largest walkable area in front of user with Orlowski Algorithm
	 */
	struct rectangleID {
		/**
		\brief rectangleID default constructor.
		 */
		rectangleID();

		/**
		\brief rectangleID constructor
		@param in_bottom: rectangle's lowest edge in Y axis
		@param in_left: rectangle's lowest edge in X axis
		@param in_top: rectangle's highest edge in Y axis
		@param in_right: rectangle's highest edge in X axis
		 */
		rectangleID(float in_bottom, float in_left, float in_top, float in_right);

		/**
		\brief Convenience function for Orlowski, if other rectangle is bigger it calls move operations
		 */
		void compare(const rectangleID& other);

		/**
		\brief Convenience function for Orlowski, determines if the max rectangle Orlowski found is big enough for a room
		 */
		bool isBigEnough();

		/**
		\brief Returns the bigger of the two rectangles in terms of area
		 */
		static const rectangleID& max(const rectangleID& a, const rectangleID& b);

		/**
		\brief The edges of the rectangle in X coords
		 */
		float left, right;

		/**
		\brief The edges of the rectangle in X coords
		 */
		float bottom, top;

		/**
		\brief The area of the rectangle
		 */
		float area;

	};

	/**
	\struct GlobalMappingInformation
	\brief Maps the currently rendered room's area in Global Map for real time obstacle detection
	\Note The room area is segmented in cells comprised of 16x16 global map cells
	 */
	struct GlobalMappingInformation
	{
		/**
		\brief GlobalMappingInformation default constructor
		 */
		GlobalMappingInformation() {
			//ilow = 0;
			//jlow = 0;
			counter = 0;
			position = sl::float2(0, 0);
		};
		GlobalMappingInformation(const GlobalMappingInformation&) = default;
		GlobalMappingInformation& operator=(const GlobalMappingInformation&) = default;

		/**
		\brief Counts if any obstacles are detected in the Global Map cells that comprise the room cell
		\Note if counter > 0 then a dynamic obstacle has emerged in the room cell
		 */
		int counter;

		/**
		\brief Position of the center of the cell in world X, Y coordinates
		 */
		sl::float2 position;
	};

}