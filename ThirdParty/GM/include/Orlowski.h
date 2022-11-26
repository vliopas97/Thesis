#pragma once

#include "BasicTypes.h"
#include "MatClass.h"

namespace GM
{
	/**
	\struct orlowski
	\brief Used for calculating the maximum walkable space in front of user with Orlowski Algorithm
	 */
	struct orlowski {
		/**
		\brief orlowski constructor.
		This function calculates walkable space in front of user with Orlowski Algorithm
		\param mat : matrix of type FLOOR to calculate the maximum open space for
		 */
		orlowski(GM::MatClass<FLOOR>& mat);
		
		/**
		\brief Holds the maximum rectangle calculated in the constructor
		 */
		rectangleID maxRectangle;
	};
}