#pragma once

#include "sl/Camera.hpp"
#include "globalMap.h"

namespace GM
{
	namespace Utilities
	{
		int initCamera(sl::Camera& zed);
		
		void transformPose(sl::Transform& pose, float tx);

	}
}