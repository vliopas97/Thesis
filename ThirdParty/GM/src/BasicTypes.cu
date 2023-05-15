#include "BasicTypes.h"

using namespace GM;

bool GM::operator==(const sl::float2 & a, const sl::float2 & b)
{
	return (a.x == b.x) && (a.y == b.y);
}

GM::limit::limit(float lowY, float lowX, float highY, float highX)
	: lowY(lowY), lowX(lowX), highY(highY), highX(highX)
{
}

GM::boundary::boundary(sl::float2 center)
	: center(center),
	dim_with_deadband(0.5 * (BOUNDARY_DIMENSIONS + DEADBAND)),
	trigger(center.y - dim_with_deadband, center.x - dim_with_deadband, center.y + dim_with_deadband, center.x + dim_with_deadband)
{
}

 __host__ __device__ void GM::boundary::update(const SLIDE & slide)
{
	switch (slide)
	{
	case SLIDE::FWD:
		center.x += BOUNDARY_DIMENSIONS;
		break;
	case SLIDE::BWD:
		center.x -= BOUNDARY_DIMENSIONS;
		break;
	case SLIDE::RGT:
		center.y += BOUNDARY_DIMENSIONS;
		break;
	case SLIDE::LFT:
		center.y -= BOUNDARY_DIMENSIONS;
		break;
	default:
		return;
	}
	trigger.lowY = center.y - dim_with_deadband;
	trigger.highY = center.y + dim_with_deadband;
	trigger.lowX = center.x - dim_with_deadband;
	trigger.highX = center.x + dim_with_deadband;
}

 //__host__ __device__ boundary & GM::boundary::operator=(const boundary & other)
 //{
	// // TODO: insert return statement here
	// center = other.center;
	// dim_with_deadband = other.dim_with_deadband;
	// trigger = other.trigger;
 //}

rectangleID::rectangleID()
{
	bottom = left = top = right = area = 0;
}

rectangleID::rectangleID(float in_bottom, float in_left, float in_top, float in_right)
	: bottom(ceil(in_bottom)), left(ceil(in_left)), top(floor(in_top)), right(floor(in_right))
{
	if ((in_bottom == in_top) || (in_left == in_right))
		area = 0;
	else
		area = (right - left) * (top - bottom);
}

void GM::rectangleID::compare(const rectangleID & other)
{
	if (other.area > area) 
	{
		*this = std::move(other);
	}
}

bool GM::rectangleID::isBigEnough()
{
	return area >= 16 && abs(top - bottom) >= 4 && abs(right - left) >= 4;
}

const rectangleID & rectangleID::max(const rectangleID & a, const rectangleID & b)
{
	return a.area < b.area ? b : a;
}