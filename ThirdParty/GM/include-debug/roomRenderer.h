#pragma once

#include "BasicTypes.h"
#include "MatClass.h"
#include "roomSegment.h"
#include "functional"

struct GlobalMap
{
	GlobalMap() : mat(1024, 1024, GM::FLOOR::UNKNOWN, GM::MatClassBase::GPUOPTIMIZATION::DISABLED)
	{
		for (int i = 387; i < 762; i++)
		{
			for (int j = 380; j < 910; j++)
			{
				if (i < 390 || i>750 || j < 390 || j>902)
				{
					mat(i, j) = GM::FLOOR::OBSTACLE;
				}
				else
				{
					mat(i, j) = GM::FLOOR::WALKABLE;
				}
			}
		}
	}

	GM::MatClass<GM::FLOOR> mat;
};

namespace GM
{
	/**
	\struct roomRenderer
	\brief Used for finding the edges of the next room to be rendered when the user enters some new area
	 */
	struct roomRenderer {
		/**
		\brief roomRenderer constructor.
		This function finds the new room and its edges
		@param location : the location of the door the user passes as he/she enters new space
		@param in_entrance: the direction of the door and hence of the user as he heads towards new space
		 */
		roomRenderer(sl::float2 location, ENTRANCE_DIR in_entrance);

		roomRenderer(sl::float2 location = sl::float2(0, 0));

		~roomRenderer() = default;

	private:
		/**
		\brief finds the area that can be used as a room and its distance to the place of entry
		@param coords : the location of the door the user passes as he/she enters new space
		@return distanceMap : a distance matrix holding the distances of the room cells to the place of entry
		 */
		void floodFill(sl::float2 coords, MatClass<int>& distanceMap);

		/**
		\brief finds room cells that can be used as portals
		@param distanceMap : a distance matrix holding the distances of the room cells to the place of entry
		 */
		void findPortals(MatClass<int>& distanceMap,const std::function<bool(roomRenderer*, int, int)>& func);

		/**
		\brief sets the coordinates of the segments to world coords
		 */	
		void alignToWorldAxis();

		/**
		\brief sets the coordinates of the segments to world coords on game load
		*/
		void alignToWorldAxisInit();

		void corridorGenerator();

		void decideSegmentsOrientation();

		void getRoomCells(int i_low, int j_low);

		static bool checkNeighbours(roomRenderer* ptr, int i, int j);

		static bool checkNeighboursInit(roomRenderer* ptr, int i, int j);

	public:

		/**
		\brief dynamic array that holds the new room's segments ordered after the constructor call
		 */
		std::vector<roomSegment> segments;

		/**
		\brief dynamic array that holds the new room's cells information
		Useful for a per frame traversal of the global map for real time obstacle detection in the room's area
		 */
		std::vector<GlobalMappingInformation> roomCells;

		//debug
		std::vector < std::vector<std::pair<FLOOR, std::vector<std::vector<float>>>>> dInfo;

	private:

		int matsize;

		MatClass<FLOOR> mat;

		sl::float2 location;

		ENTRANCE_DIR entrance;

	public:

		static GlobalMap Global;

	};

}

