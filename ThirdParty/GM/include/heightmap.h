#pragma once

#include "MatClass.h"

namespace GM
{
	using typeofHeight = float;

	/**
	\class HeightMapClass
	\brief MatClass type of array that can perform shifting operations
	 */
	class HeightMapClass : public MatClass<typeofHeight>
	{

	public:

		using MatClass<typeofHeight>::MatClass;

		/**
		\brief Shifts the array on a specific direction
		@param slide : specify the shifting direction
		 */
		__host__ __device__ void Slide(SLIDE slide);

	private:

		__host__ void SlideBackwardHost();

		__host__ void SlideForwardHost();

		__host__ void SlideRightwardHost();

		__host__ void SlideLeftwardHost();

	public:
		/**
		\brief The number of rows or columns to be cleared after Slide
		 */
		unsigned int offset = 50;

		friend class GlobalMapDevice;

	};

}