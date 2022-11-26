#pragma once

#include "sl/Camera.hpp"
#include "BasicTypes.h"

namespace GM
{
	/**
	\struct roomSegment
	\brief Contains data for room's edges seeing them as line segments, useful for rendering
	\Note The room area is segmented in cells comprised of 16x16 global map cells
	 */
	struct roomSegment {

		/**
		\enum SegmentType
		\brief Lists the different types of room edges
		 */
		enum class SegmentType {
			WALL = 0,
			PORTAL,/**< Means segments separates the room with unmapped area, could be used as door or as wall*/
			DOOR
		};
		
		/**
		\brief roomSegment constructor
		@param pointA : first point of the segment
		@param pointB : second point of the segment
		 */
		roomSegment(const sl::float2& pointA, const sl::float2& pointB);

		/**
		\brief roomSegment constructor
		@param pointA : first point of the segment
		@param pointB : second point of the segment
		@param neighbour : floor variable of neighbour to decide segment's type
		 */
		roomSegment(const sl::float2& pointA, const sl::float2& pointB, const FLOOR& neighbour); 

		/**
		\brief roomSegment constructor
		@param pointA : first point of the segment
		@param pointB : second point of the segment
		@param neighbour : segment type for the object
		 */
		roomSegment(const sl::float2& pointA, const sl::float2& pointB, const SegmentType& segmentType);

		roomSegment(const roomSegment& other) = default;

		roomSegment& operator=(const roomSegment& other) = default;

		/**
		\brief Checks if the room segment neighbors a second segment
		@param other : the other segment
		 */
		bool areNeighbours(const roomSegment& other) const;

		/**
		\brief Checks if two segments are one the same line
		@param other : the other segment
		 */
		bool sameLine(const roomSegment& other);

		/**
		\brief Merges two segments
		@param other : the other segment
		\Note The function checks if the segments have a common point before proceeding
		 */
		void mergeSegment(const roomSegment& other);

		/**
		\brief Calculates the center point of the segment
		@return: The center point of the segment
		 */
		sl::float2 center() const;

		SegmentType segmentType;

		/**
		\brief The points of the segments
		*/
		sl::float2 pointA, pointB;

		/**
		\brief Orientation of the segment. Useful for rendering purposes
		*/
		ENTRANCE_DIR direction = ENTRANCE_DIR::WALL;

	};
}