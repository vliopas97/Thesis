#include "roomSegment.h"

using namespace GM;

bool GM::operator==(const roomSegment & a, const roomSegment & b)
{
	return ((a.pointA == b.pointA) || (a.pointA == b.pointB)) && a.center() == b.center();
}

roomSegment::roomSegment(const sl::float2& pointA, const sl::float2& pointB)
	: pointA(pointA), pointB(pointB), segmentType(SegmentType::WALL)
{
}

roomSegment::roomSegment(const sl::float2& pointA, const sl::float2& pointB, const FLOOR & neighbour)
	: pointA(pointA), pointB(pointB)
{
	segmentType = neighbour == FLOOR::OBSTACLE ? SegmentType::WALL : SegmentType::PORTAL;
}

roomSegment::roomSegment(const sl::float2& pointA, const sl::float2& pointB, const SegmentType & segmentType)
	: pointA(pointA), pointB(pointB), segmentType(segmentType)
{
}

bool GM::roomSegment::areNeighbours(const roomSegment & other) const
{
	if (pointA == other.pointA || pointA == other.pointB || pointB == other.pointA || pointB == other.pointB)
		if (center() == other.center())
			return false;
	return true;
}

bool GM::roomSegment::sameLine(const roomSegment & other)
{
	sl::float2 AB(this->pointB - this->pointA);
	sl::float2 CD(other.pointB - other.pointA);
	return abs(AB.dot(AB, CD) / ((pointB.distance(this->pointA, this->pointB)) *(pointB.distance(other.pointA, other.pointB))) - 1) < 0.00001;
}

void roomSegment::mergeSegment(const roomSegment & other)
{
	_ASSERT(sameLine(other));
	auto d1 = pointA.distance(pointA, other.pointA);
	auto d2 = pointA.distance(pointA, other.pointB);
	auto d3 = pointA.distance(pointB, other.pointA);
	auto d4 = pointA.distance(pointB, other.pointB);
	auto d = std::max({ d1, d2, d3, d4 });
	if (d == d1)
	{
		pointA = pointA;
		pointB = other.pointA;
	}
	else if (d == d2)
	{
		pointA = pointA;
		pointB = other.pointB;
	}
	else if (d == d3)
	{
		pointA = pointB;
		pointB = other.pointA;
	}
	else
	{
		pointA = pointB;
		pointB = other.pointB;
	}

	//if (pointA == other.pointA)
	//{
	//	pointA = pointB;
	//	pointB = other.pointB;
	//}
	//else if (pointA == other.pointB)
	//{
	//	pointA = pointB;
	//	pointB = other.pointA;
	//}
	//else if (pointB == other.pointA)
	//{
	//	pointA = pointA;
	//	pointB = other.pointB;
	//}
	//else if (pointB == other.pointB)
	//{
	//	pointA = pointA;
	//	pointB = other.pointA;
	//}
	//else
	//{
	//	auto d1 = pointA.distance(pointA, other.pointA);
	//	auto d2 = pointA.distance(pointA, other.pointB);
	//	auto d3 = pointA.distance(pointB, other.pointA);
	//	auto d4 = pointA.distance(pointB, other.pointB);
	//	auto d = std::max({ d1, d2, d3, d4 });
	//
	//}


}

sl::float2 GM::roomSegment::center() const
{
	return (this->pointA + this->pointB) / 2.0;
}
