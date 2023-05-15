#include "utilities.h"

using namespace sl;

int GM::Utilities::initCamera(sl::Camera& zed)
{
	// Set configuration parameters
	//sl::InitParameters init_params;
	//init_params.coordinate_system = COORDINATE_SYSTEM::LEFT_HANDED_Z_UP; // Use a right-handed Y-up coordinate system
	//init_params.coordinate_units = UNIT::CENTIMETER; // Set units in meters
	////init_params.sensors_required = true;
	//init_params.depth_mode = DEPTH_MODE::PERFORMANCE;
	//init_params.camera_resolution = RESOLUTION::HD720;
	//init_params.input.setFromSVOFile("C:/Users/Vangelis Liopas/Documents/ZED/HD1080_SN10027804_11-21-45.svo");

	//// Open the camera
	//ERROR_CODE err = zed.open(init_params);
	//if (err != ERROR_CODE::SUCCESS) {
	//	std::cout << "Error " << err << ", exit program.\n";
	//	return -1;
	//}

	//// Enable positional tracking with default parameters
	//PositionalTrackingParameters tracking_parameters;
	////tracking_parameters.initial_world_transform.setTranslation(Translation(0, 0, 60));
	//tracking_parameters.enable_area_memory = true;
	//tracking_parameters.enable_pose_smoothing = true;

	//err = zed.enablePositionalTracking(tracking_parameters);
	//if (err != ERROR_CODE::SUCCESS)
	//	return -1;
	return 0;
}

void GM::Utilities::transformPose(sl::Transform& pose, float tx) {
	Transform transform_;
	transform_.setIdentity();
	// Translate the tracking frame by tx along the X axis  
	transform_.tx = tx;
	// Pose(new reference frame) = M.inverse() * Pose (camera frame) * M, where M is the transform between two frames
	pose = Transform::inverse(transform_) * pose * transform_;
}

