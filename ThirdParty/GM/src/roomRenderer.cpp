#include "roomRenderer.h"
#include "Orlowski.h"
#include "globalMap.h"
#include <numeric>
#include <utility>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace GM;

//for debug purposes only - predefined input for global map
#if DEBUG 1
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

GlobalMap Global;
namespace
{
	void copyGlobalMap(sl::float2 location, ENTRANCE_DIR entrance, GM::MatClass<FLOOR>& mat, int& ilow, int& jlow)
	{
		float l = 2048;
		int row = static_cast<int>(floorf(((location.x + l) / 4.))) % MAP_DIMENSIONS;
		int column = static_cast<int>(floorf(((location.y + l) / 4.))) % MAP_DIMENSIONS;

		switch (entrance)
		{
		case GM::ENTRANCE_DIR::X_UP:
			ilow = row;
			jlow = column - 127;
			for (int i = 0; i < 256; i++)
			{
				for (int j = 0; j < 256; j++)
				{
					mat(i, j) = Global.mat(ilow + i, jlow + j);
				}
			}
			break;
		case GM::ENTRANCE_DIR::X_DOWN:
			ilow = row - 255;
			jlow = column - 127;
			for (int i = 0; i < 256; i++)
			{
				for (int j = 0; j < 256; j++)
				{
					mat(255 - i, 255 - j) = Global.mat(ilow + i, jlow + j);
				}
			}
			break;
		case GM::ENTRANCE_DIR::Y_UP:
			ilow = row - 127;
			jlow = column;
			for (int i = 0; i < 256; i++)
			{
				for (int j = 0; j < 256; j++)
				{
					mat(j, 255 - i) = Global.mat(ilow + i, jlow + j);
				}
			}
			break;
		case GM::ENTRANCE_DIR::Y_DOWN:
			ilow = row - 127;
			jlow = column - 255;
			for (int i = 0; i < 256; i++)
			{
				for (int j = 0; j < 256; j++)
				{
					mat(255 - j, i) = Global.mat(ilow + i, jlow + j);
				}
			}
			break;
		default:
			break;
		}

	}

	void copyGlobalMapInit(sl::float2 location, GM::MatClass<FLOOR>& mat, int& ilow, int& jlow)
	{
		float l = 2048;
		int row = static_cast<int>(floorf((location.x + l / 4.))) % MAP_DIMENSIONS;
		int column = static_cast<int>(floorf((location.y + l / 4.))) % MAP_DIMENSIONS;

		ilow = row - 127;
		jlow = column - 127;

		for (int i = 0; i < 256; i++)
		{
			for (int j = 0; j < 256; j++)
			{
				mat(i, j) = Global.mat(ilow + i, jlow + j);
			}
		}
	}
}
#endif // DEBUG 1

std::vector<std::vector<std::tuple<float, float, unsigned int>>> GM::roomRenderer::msInfo;


std::ostream& operator<<(std::ostream& stream, const FLOOR& floor)
{
	switch (floor)
	{
	case(FLOOR::UNKNOWN):
		stream << "U";
		return stream;
	case(FLOOR::WALKABLE):
		stream << "W";
		return stream;
	case(FLOOR::OBSTACLE):
		stream << "O";
		return stream;
	case(FLOOR::PORTAL):
		stream << "P";
		return stream;
	case(FLOOR::ROOM):
		stream << "R";
		return stream;
	default:
		stream << "X";
		return stream;
	}
}

template<typename T>
void print(MatClass<T>& mat)
{
	for (int i = mat.rows - 1; i >= 0; i--)
	{
		std::cout << " ";
		for (int j = 0; j < mat.columns; j++)
			//printf("%2d ", mat(i, c));
			std::cout << " " << mat(i, j) << " ";
		std::cout << "\n";
	}
	puts("");
}

void presentresult(MatClass<FLOOR>& mat, const rectangleID& rectangle)
{
	for (size_t i = rectangle.left; i < rectangle.right; i++)
		for (size_t j = rectangle.bottom; j < rectangle.top; j++)
		{
			mat(j, i) = FLOOR::ROOM;
		}
}

namespace
{
	class pixelBlock {
	public:
		virtual ~pixelBlock() = default;

		virtual void setVisited(MatClass<FLOOR>& mat) = 0;

		virtual void expand(MatClass<FLOOR>& mat, std::vector<std::unique_ptr<pixelBlock>>& stack) = 0;
	protected:

		static void appendToStack(std::vector<std::unique_ptr<pixelBlock>>& stack, std::unique_ptr<pixelBlock>&& ptr)
		{
			stack.emplace_back(std::move(ptr));
		}

		static bool expansionCheck(const sl::float2& point1, const sl::float2& point2, MatClass<FLOOR>& mat)
		{
			return !((point1.x < 0 || point1.x >= mat.rows) || (point2.x < 0 || point2.x >= mat.rows) ||
				(point1.y < 0 || point1.y >= mat.columns) || (point2.y < 0 || point2.y >= mat.columns) ||
				((mat(point1.x, point1.y) == FLOOR::OBSTACLE || mat(point1.x, point1.y) == FLOOR::UNKNOWN) ||
				(mat(point2.x, point2.y) == FLOOR::OBSTACLE || mat(point2.x, point2.y) == FLOOR::UNKNOWN)) ||
				(mat(point1.x, point1.y) == FLOOR::ROOM && mat(point2.x, point2.y) == FLOOR::ROOM) ||
				(mat(point1.x, point1.y) == FLOOR::PORTAL && mat(point2.x, point2.y) == FLOOR::PORTAL));
		}

	};

	class pixelBlock_4 : public pixelBlock {
	public:
		pixelBlock_4();

		pixelBlock_4(sl::float2 a, sl::float2 b, sl::float2 c, sl::float2 d);

		static void expandInternal(sl::float2 closePoint1, sl::float2 closePoint2, sl::float2 farPoint1, sl::float2 farPoint2, MatClass<FLOOR>& mat, std::vector<std::unique_ptr<pixelBlock>>& stack);

		bool checkIfEntranceReached(MatClass<int>& distanceMap);

		pixelBlock_4 findClosest(MatClass<FLOOR>& mat, MatClass<int>& distanceMap);

		void setVisited(MatClass<FLOOR>& mat) override;

		void expand(MatClass<FLOOR>& mat, std::vector<std::unique_ptr<pixelBlock>>& stack) override;

	private:
		static int min(int a, int b, int c, int d);

		void expandUp(MatClass<FLOOR>& mat, std::vector<std::unique_ptr<pixelBlock>>& stack);

		void expandDown(MatClass<FLOOR>& mat, std::vector<std::unique_ptr<pixelBlock>>& stack);

		void expandLeft(MatClass<FLOOR>& mat, std::vector<std::unique_ptr<pixelBlock>>& stack);

		void expandRight(MatClass<FLOOR>& mat, std::vector<std::unique_ptr<pixelBlock>>& stack);

		int findClosestUp(MatClass<FLOOR>& mat, MatClass<int>& distanceMap);

		int findClosestDown(MatClass<FLOOR>& mat, MatClass<int>& distanceMap);

		int findClosestLeft(MatClass<FLOOR>& mat, MatClass<int>& distanceMap);

		int findClosestRight(MatClass<FLOOR>& mat, MatClass<int>& distanceMap);

		sl::float2 bLeft, bRight, uLeft, uRight;
	public:
		bool expansionFlag = true;
	};

	class pixelBlock_2 : public pixelBlock {
	public:

		pixelBlock_2(sl::float2 a, sl::float2 b);

		bool checkIfEntranceReached(MatClass<int>& distanceMap);

		pixelBlock_4 findClosest(MatClass<FLOOR>& mat, MatClass<int>& distanceMap);

		void expand(MatClass<FLOOR>& mat, std::vector<std::unique_ptr<pixelBlock>>& stack) override;

	private:
		static int min(int a, int b);

		void setVisited(MatClass<FLOOR>& mat) override;

		int findClosestUp(MatClass<FLOOR>& mat, MatClass<int>& distanceMap);

		int findClosestDown(MatClass<FLOOR>& mat, MatClass<int>& distanceMap);

		int findClosestLeft(MatClass<FLOOR>& mat, MatClass<int>& distanceMap);

		int findClosestRight(MatClass<FLOOR>& mat, MatClass<int>& distanceMap);

		sl::float2 pointa, pointb;
		bool horizontal;

	public:
		bool expansionFlag = true;
	};

		pixelBlock_4::pixelBlock_4() = default;

		pixelBlock_4::pixelBlock_4(sl::float2 a, sl::float2 b, sl::float2 c, sl::float2 d)
		{
			float x = std::min(a.x, std::min(b.x, std::min(c.x, d.x)));
			float y = std::min(a.y, std::min(b.y, std::min(c.y, d.y)));
			bLeft = sl::float2(x, y);
			bRight = sl::float2(x, y + 1);
			uLeft = sl::float2(x + 1, y);
			uRight = sl::float2(x + 1, y + 1);
		}

		void pixelBlock_4::expandInternal(sl::float2 closePoint1, sl::float2 closePoint2, sl::float2 farPoint1, sl::float2 farPoint2, MatClass<FLOOR>& mat, std::vector<std::unique_ptr<pixelBlock>>& stack)
		{
			if (pixelBlock::expansionCheck(closePoint1, closePoint2, mat))
				if (pixelBlock::expansionCheck(farPoint1, farPoint2, mat))
					pixelBlock::appendToStack(stack, std::make_unique<pixelBlock_4>(closePoint1, closePoint2, farPoint1, farPoint2));
				else
					pixelBlock::appendToStack(stack, std::make_unique<pixelBlock_2>(closePoint1, closePoint2));
		}

		bool pixelBlock_4::checkIfEntranceReached(MatClass<int>& distanceMap)
		{
			return ((distanceMap(bLeft.x, bLeft.y) == 0 ? 1 : 0) + (distanceMap(uLeft.x, uLeft.y) == 0 ? 1 : 0)
				+ (distanceMap(bRight.x, bRight.y) == 0 ? 1 : 0) + (distanceMap(uRight.x, uRight.y) == 0 ? 1 : 0)) == 2;
		}

		pixelBlock_4 pixelBlock_4::findClosest(MatClass<FLOOR>& mat, MatClass<int>& distanceMap)
		{
			int d1 = findClosestUp(mat, distanceMap);
			int d2 = findClosestDown(mat, distanceMap);
			int d3 = findClosestLeft(mat, distanceMap);
			int d4 = findClosestRight(mat, distanceMap);
			int index = pixelBlock_4::min(d1, d2, d3, d4);
			if (d1 < 0 && d2 < 0 && d3 < 0 && d4 < 0) {
				expansionFlag = false;
				return pixelBlock_4();
			}
			else if (index == 1) 
			{
				return pixelBlock_4(uLeft, uRight, sl::float2(uLeft.x + 1, uLeft.y), sl::float2(uRight.x + 1, uRight.y));
			}
			else if (index == 2) 
			{
				return pixelBlock_4(bLeft, bRight, sl::float2(bLeft.x - 1, bLeft.y), sl::float2(bRight.x - 1, bRight.y));
			}
			else if (index == 3) 
			{
				return pixelBlock_4(bLeft, uLeft, sl::float2(bLeft.x, bLeft.y - 1), sl::float2(uLeft.x, uLeft.y - 1));
			}
			else /*if (index == 4)*/ 
			{
				return pixelBlock_4(bRight, uRight, sl::float2(bRight.x, bRight.y + 1), sl::float2(uRight.x, uRight.y + 1));
			}
		}

		void pixelBlock_4::setVisited(MatClass<FLOOR>& mat)
		{
			mat(bLeft.x, bLeft.y) = mat(bRight.x, bRight.y) = mat(uLeft.x, uLeft.y) = mat(uRight.x, uRight.y) = FLOOR::ROOM;
		}

		int pixelBlock_4::min(int a, int b, int c, int d)
		{
			a = a < 0 ? 100 : a;
			b = b < 0 ? 100 : b;
			c = c < 0 ? 100 : c;
			d = d < 0 ? 100 : d;
			int min = std::min({ a, b, c, d });
			if (a == min)
				return 1;
			else if (b == min)
				return 2;
			else if (c == min)
				return 3;
			else /*if (d == min)*/
				return 4;
		}

		void pixelBlock_4::expandUp(MatClass<FLOOR>& mat, std::vector<std::unique_ptr<pixelBlock>>& stack)
		{
			sl::float2 newpx1 = sl::float2(uLeft.x + 1, uLeft.y);
			sl::float2 newpx2 = sl::float2(uRight.x + 1, uRight.y);
			if (pixelBlock::expansionCheck(newpx1, newpx2, mat))
				pixelBlock::appendToStack(stack, std::make_unique<pixelBlock_4>(uLeft, uRight, newpx1, newpx2));
		}

		void pixelBlock_4::expandDown(MatClass<FLOOR>& mat, std::vector<std::unique_ptr<pixelBlock>>& stack)
		{
			sl::float2 newpx1 = sl::float2(bLeft.x - 1, bLeft.y);
			sl::float2 newpx2 = sl::float2(bRight.x - 1, bRight.y);
			if (pixelBlock::expansionCheck(newpx1, newpx2, mat))
				pixelBlock::appendToStack(stack, std::make_unique<pixelBlock_4>(bLeft, bRight, newpx1, newpx2));
		}

		void pixelBlock_4::expandLeft(MatClass<FLOOR>& mat, std::vector<std::unique_ptr<pixelBlock>>& stack)
		{
			sl::float2 newpx1 = sl::float2(bLeft.x, bLeft.y - 1);
			sl::float2 newpx2 = sl::float2(uLeft.x, uLeft.y - 1);
			if (pixelBlock::expansionCheck(newpx1, newpx2, mat))
				pixelBlock::appendToStack(stack, std::make_unique<pixelBlock_4>(bLeft, uLeft, newpx1, newpx2));
		}

		void pixelBlock_4::expandRight(MatClass<FLOOR>& mat, std::vector<std::unique_ptr<pixelBlock>>& stack)
		{
			sl::float2 newpx1 = sl::float2(bRight.x, bRight.y + 1);
			sl::float2 newpx2 = sl::float2(uRight.x, uRight.y + 1);
			if (pixelBlock::expansionCheck(newpx1, newpx2, mat))
				pixelBlock::appendToStack(stack, std::make_unique<pixelBlock_4>(bRight, bRight, newpx1, newpx2));
		}

		void pixelBlock_4::expand(MatClass<FLOOR>& mat, std::vector<std::unique_ptr<pixelBlock>>& stack)
		{
			expandUp(mat, stack);
			expandDown(mat, stack);
			expandLeft(mat, stack);
			expandRight(mat, stack);
		}

		int pixelBlock_4::findClosestUp(MatClass<FLOOR>& mat, MatClass<int>& distanceMap)
		{
			sl::float2 newpx1 = sl::float2(uLeft.x + 1, uLeft.y);
			sl::float2 newpx2 = sl::float2(uRight.x + 1, uRight.y);
			if (pixelBlock::expansionCheck(newpx1, newpx2, mat)) {
				return std::min(distanceMap(newpx1.x, newpx1.y), distanceMap(newpx2.x, newpx2.y));
			}
			return -1;
		}

		int pixelBlock_4::findClosestDown(MatClass<FLOOR>& mat, MatClass<int>& distanceMap)
		{
			sl::float2 newpx1 = sl::float2(bLeft.x - 1, bLeft.y);
			sl::float2 newpx2 = sl::float2(bRight.x - 1, bRight.y);
			if (pixelBlock::expansionCheck(newpx1, newpx2, mat)) {
				return std::min(distanceMap(newpx1.x, newpx1.y), distanceMap(newpx2.x, newpx2.y));
			}
			return -1;
		}

		int pixelBlock_4::findClosestLeft(MatClass<FLOOR>& mat, MatClass<int>& distanceMap)
		{
			sl::float2 newpx1 = sl::float2(bLeft.x, bLeft.y - 1);
			sl::float2 newpx2 = sl::float2(uLeft.x, uLeft.y - 1);
			if (pixelBlock::expansionCheck(newpx1, newpx2, mat)) {
				return std::min(distanceMap(newpx1.x, newpx1.y), distanceMap(newpx2.x, newpx2.y));
			}
			return -1;
		}

		int pixelBlock_4::findClosestRight(MatClass<FLOOR>& mat, MatClass<int>& distanceMap)
		{
			sl::float2 newpx1 = sl::float2(bRight.x, bRight.y + 1);
			sl::float2 newpx2 = sl::float2(uRight.x, uRight.y + 1);
			if (pixelBlock::expansionCheck(newpx1, newpx2, mat)) {
				return std::min(distanceMap(newpx1.x, newpx1.y), distanceMap(newpx2.x, newpx2.y));
			}
			return -1;
		}

		pixelBlock_2::pixelBlock_2(sl::float2 a, sl::float2 b)
			:pointa(a), pointb(b)
		{
			horizontal = abs(a.x - b.x) == 0;
		}

		bool pixelBlock_2::checkIfEntranceReached(MatClass<int>& distanceMap)
		{
			return distanceMap(pointa.x, pointa.y) == 0 && distanceMap(pointb.x, pointb.y) ==0;
		}

		pixelBlock_4 pixelBlock_2::findClosest(MatClass<FLOOR>& mat, MatClass<int>& distanceMap)
		{
			if (horizontal)
			{
				int d1 = findClosestUp(mat, distanceMap);
				int d2 = findClosestDown(mat, distanceMap);
				int min = pixelBlock_2::min(d1, d2);

				if (d1 < 0 && d2 < 0)
				{
					expansionFlag = false;
					return pixelBlock_4();
				}
				else if (min == 1)
				{
					return pixelBlock_4(pointa, pointb, sl::float2(pointa.x + 1, pointa.y), sl::float2(pointb.x + 1, pointb.y));
				}
				else/* if (min == 2)*/
				{
					return pixelBlock_4(pointa, pointb, sl::float2(pointa.x - 1, pointa.y), sl::float2(pointb.x - 1, pointb.y));
				}
			}
			else
			{
				int d1 = findClosestLeft(mat, distanceMap);
				int d2 = findClosestRight(mat, distanceMap);
				int min = pixelBlock_2::min(d1, d2);

				if (d1 < 0 && d2 < 0)
				{
					expansionFlag = false;
					return pixelBlock_4();
				}
				else if (min == 1)
				{
					return pixelBlock_4(pointa, pointb, sl::float2(pointa.x, pointa.y - 1), sl::float2(pointb.x, pointb.y - 1));
				}
				else /*if (min == 2)*/
				{
					return pixelBlock_4(pointa, pointb, sl::float2(pointa.x, pointa.y + 1), sl::float2(pointb.x, pointb.y + 1));
				}
			}
		}

		int pixelBlock_2::min(int a, int b)
		{
			a = a < 0 ? 100 : a;
			b = b < 0 ? 100 : b;
			int min = std::min(a, b);
			if (a == min)
				return 1;
			else /*if (b == min)*/
				return 2;
		}

		void pixelBlock_2::expand(MatClass<FLOOR>& mat, std::vector<std::unique_ptr<pixelBlock>>& stack)
		{
		}

		void pixelBlock_2::setVisited(MatClass<FLOOR>& mat)
		{
			mat(pointa.x, pointa.y) = mat(pointb.x, pointb.y) = FLOOR::ROOM;
		}

		int pixelBlock_2::findClosestUp(MatClass<FLOOR>& mat, MatClass<int>& distanceMap)
		{
			sl::float2 newpx1 = sl::float2(pointa.x + 1, pointa.y);
			sl::float2 newpx2 = sl::float2(pointb.x + 1, pointb.y);
			if (pixelBlock::expansionCheck(newpx1, newpx2, mat)) {
				return std::min(distanceMap(newpx1.x, newpx1.y), distanceMap(newpx2.x, newpx2.y));
			}
			return -1;
		}

		int pixelBlock_2::findClosestDown(MatClass<FLOOR>& mat, MatClass<int>& distanceMap)
		{
			sl::float2 newpx1 = sl::float2(pointa.x - 1, pointa.y);
			sl::float2 newpx2 = sl::float2(pointb.x - 1, pointb.y);
			if (pixelBlock::expansionCheck(newpx1, newpx2, mat)) {
				return std::min(distanceMap(newpx1.x, newpx1.y), distanceMap(newpx2.x, newpx2.y));
			}
			return -1;
		}

		int pixelBlock_2::findClosestLeft(MatClass<FLOOR>& mat, MatClass<int>& distanceMap)
		{
			sl::float2 newpx1 = sl::float2(pointa.x, pointa.y - 1);
			sl::float2 newpx2 = sl::float2(pointb.x, pointb.y - 1);
			if (pixelBlock::expansionCheck(newpx1, newpx2, mat)) {
				return std::min(distanceMap(newpx1.x, newpx1.y), distanceMap(newpx2.x, newpx2.y));
			}
			return -1;
		}

		int pixelBlock_2::findClosestRight(MatClass<FLOOR>& mat, MatClass<int>& distanceMap)
		{
			sl::float2 newpx1 = sl::float2(pointa.x, pointa.y + 1);
			sl::float2 newpx2 = sl::float2(pointb.x, pointb.y + 1);
			if (pixelBlock::expansionCheck(newpx1, newpx2, mat)) {
				return std::min(distanceMap(newpx1.x, newpx1.y), distanceMap(newpx2.x, newpx2.y));
			}
			return -1;
		}

		void mergeSegmentSequence(std::vector<roomSegment>& segment, int first, int last)
		{
			if (first == last || first >= segment.size() - 1)
				return;

			segment[first].mergeSegment(segment[last]);
			segment.erase(segment.begin() + first + 1, segment.begin() + last + 1);
		}

		void appendSegment(std::vector<roomSegment>& segments, const roomSegment& segment)
		{
			if (std::find_if(segments.begin(), segments.end(),
				[&segment](const roomSegment& a) {
				return ((a.pointA == segment.pointA) && (a.pointB == segment.pointB)) ||
					((a.pointA == segment.pointB) && (a.pointB == segment.pointA));
			}) == segments.end())
				segments.emplace_back(segment);
		}

		void sortSegmentInternal(int index, sl::float2 point, std::vector<roomSegment>& segments, std::vector<roomSegment>& sorted)
		{			
			sorted.emplace_back(segments[index]);
			segments.erase(segments.begin() + index);
			if (segments.empty())
				return;

			auto newindex = std::find_if(segments.begin(), segments.end(), [&point](const roomSegment& segment) {
				return segment.pointA == point || segment.pointB == point;
			}) - segments.begin();

			if (newindex == segments.size()) return;

			if (segments[newindex].pointA == point) {
				sortSegmentInternal(newindex, segments[newindex].pointB, segments, sorted);
			}
			else {
				sortSegmentInternal(newindex, segments[newindex].pointA, segments, sorted);
			}
		}

		void sortSegment(std::vector<roomSegment>& segments)
		{
			std::vector<roomSegment> sorted;
			sortSegmentInternal(0, segments[0].pointB, segments, sorted);
			segments.clear();
			segments = sorted;
		}

		void decidePortals(std::vector<roomSegment>& segment)
		{
			using SegmentType = roomSegment::SegmentType;
			int i = 0;
			while (i < segment.size()) {
				const auto& ref = segment[i].segmentType;
				

				if (ref == roomSegment::SegmentType::PORTAL) {
					int j = 1;
					while (i + j < segment.size() && segment[i + j].segmentType == ref && segment[i].sameLine(segment[i + j]) && j <4) {
						j++;
					}

					if (j == 1) {
						segment[i].segmentType = SegmentType::WALL;
						if (i > 0 && segment[i - 1].segmentType == SegmentType::WALL) {
							i--;
						}
					}
					else if (j == 2) {
						mergeSegmentSequence(segment, i, i + 1);
						i++;
					}
					else if (j == 3) {
						//merge 2 first or two last
						//set the other to wall if first is set the i = i-1 and repeat
						int randomNum = ((double)rand() / (RAND_MAX));
						if (randomNum <= 0.5) {
							mergeSegmentSequence(segment, i, i + 1);
							i += 1;
							if (i < segment.size())
								segment[i].segmentType = SegmentType::WALL;
						}
						else {
							mergeSegmentSequence(segment, i + 1, i + 2);
							segment[i].segmentType = SegmentType::WALL;
							if (i - 1 >= 0 && segment[i - 1].segmentType == SegmentType::WALL)
							{
								mergeSegmentSequence(segment, i - 1, i);
							}
							i--;
						}
					}
					else {
						//take first four merge the middle ones
						mergeSegmentSequence(segment, i + 1, i + 2);
						segment[i].segmentType = SegmentType::WALL;
						if (i - 1 >= 0 && segment[i - 1].segmentType == SegmentType::WALL && segment[i].sameLine(segment[i-1]))
						{
							mergeSegmentSequence(segment, i - 1, i);
							i--;
						}
						i += 2;
						if (i < segment.size())
						{
							segment[i].segmentType = SegmentType::WALL;
						}
					}
				}

				else {
					int j = 1;
					while (i + j < segment.size() && segment[i + j].segmentType == ref && segment[i].sameLine(segment[i + j]))
					{
						j++;
					}
					if (j != 1) 
						mergeSegmentSequence(segment, i, i + j - 1);
					i++;
				}
			}
		}

	void erodeInternal(MatClass<FLOOR>& mat, sl::float2 elementpos)
	{
		static const int erosion = 5;
		for (size_t i = elementpos.x - erosion < 0 ? 0 : elementpos.x - erosion; i < mat.rows && i <= elementpos.x + erosion; i++) {
			for (size_t j = elementpos.y - erosion < 0 ? 0 : elementpos.y - erosion; j < mat.columns && j <= elementpos.y + erosion; j++) {
					mat(i, j) = FLOOR::OBSTACLE;
			}
		}
	}

	void erodeInternal(MatClass<float>& mat, sl::float2 elementpos)
	{
		static const int erosion = 5;
		for (size_t i = elementpos.x - erosion < 0 ? 0 : elementpos.x - erosion; i < mat.rows && i <= elementpos.x + erosion; i++)
		{
			for (size_t j = elementpos.y - erosion < 0 ? 0 : elementpos.y - erosion; j < mat.columns && j <= elementpos.y + erosion; j++)
			{
				mat(i, j) = 100.f;
			}
		}
	}

	void erode(MatClass<FLOOR>& mat)
	{
		std::vector<sl::float2> obstacles;
		for (size_t i = 0; i < mat.rows; i++) {
			for (size_t j = 0; j < mat.columns; j++) {
				if (mat(i, j) == FLOOR::OBSTACLE)
					obstacles.emplace_back(sl::float2(i, j));
			}
		}
		for (auto& o : obstacles)
			erodeInternal(mat, o);
	}

	void erode(MatClass<float>& mat)
	{
		std::vector<sl::float2> obstacles;
		for (size_t i = 0; i < mat.rows; i++)
		{
			for (size_t j = 0; j < mat.columns; j++)
			{
				if (mat(i, j) > THRESHOLD)
					obstacles.emplace_back(sl::float2(i, j));
			}
		}
	}

	void downsample(MatClass<FLOOR>& tempmat, MatClass<FLOOR>& mat)
	{
		_ASSERT(mat.rows == tempmat.rows / downsampleFactor && mat.columns == tempmat.columns / downsampleFactor);
		for (size_t i = 0; i < tempmat.rows; i++) {
			auto newrow = floor(i / downsampleFactor);
			for (size_t j = 0; j < tempmat.columns; j++) {
				auto newcolumn = floor(j / downsampleFactor);
				if (mat(newrow, newcolumn) == FLOOR::DOWNSAMPLED_INIT)
					mat(newrow, newcolumn) = tempmat(i, j);
				else if (mat(newrow, newcolumn) == FLOOR::UNKNOWN && tempmat(i, j) == FLOOR::OBSTACLE)
					mat(newrow, newcolumn) = FLOOR::OBSTACLE;
				else if (mat(newrow, newcolumn) == FLOOR::WALKABLE && tempmat(i, j) != FLOOR::WALKABLE) {
					mat(newrow, newcolumn) = tempmat(i, j);
				}
			}
		}
	}

	double phi(double x)
	{
		// constants
		double a1 = 0.254829592;
		double a2 = -0.284496736;
		double a3 = 1.421413741;
		double a4 = -1.453152027;
		double a5 = 1.061405429;
		double p = 0.3275911;

		// Save the sign of x
		int sign = 1;
		if (x < 0)
			sign = -1;
		x = fabs(x) / sqrt(2.0);

		// A&S formula 7.1.26
		double t = 1.0 / (1.0 + p * x);
		double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

		return 0.5 * (1.0 + sign * y);
	}

	void downsample(MatClass<float>& tempmat, MatClass<FLOOR>& mat)
	{
		_ASSERT(mat.rows == tempmat.rows / downsampleFactor && mat.columns == tempmat.columns / downsampleFactor);
		std::vector< std::vector<std::tuple<float, float, unsigned int>>> NormalDist(mat.rows,
			std::vector<std::tuple<float, float, unsigned int>>(mat.columns, std::make_tuple(0, 0, 0))); //mean - standard deviation - unmapped instances
		
		for (size_t i = 0; i < tempmat.rows; i++)
		{
			auto newrow = floor(i / downsampleFactor);
			for (size_t j = 0; j < tempmat.columns; j++)
			{
				auto newcolumn = floor(j / downsampleFactor);
				if (tempmat(i, j) == -1) 
				{
					std::get<2>(NormalDist[newrow][newcolumn])++;//sum of unmapped pieces
				}
				else
				{
					std::get<0>(NormalDist[newrow][newcolumn]) += tempmat(i, j);
				}
			}
		}

		int row = 0;
		int col = 0;

		for (size_t i = 0; i < NormalDist.size(); i++)
		{
			for (size_t j = 0; j < NormalDist.size(); j++)
			{
				auto& element = NormalDist[i][j];
				if (std::get<2>(element) == mat.rows * mat.columns)
					continue;

				std::get<0>(element) /= (mat.rows * mat.columns - std::get<2>(element));
				auto sum = 0;

				for (size_t row = i * downsampleFactor; row < downsampleFactor * i + downsampleFactor; row++)
				{
					for (size_t col = j * downsampleFactor; col < downsampleFactor * j + downsampleFactor; col++)
					{
						if (tempmat(row, col) != -1)
							sum += (tempmat(row, col) - std::get<0>(element)) * (tempmat(row, col) - std::get<0>(element));
					}
				}
				std::get<1>(element) = sqrt(sum / (static_cast<float>(mat.rows * mat.columns) - static_cast<float>(std::get<2>(element))));//sigma
			}
		}

		for (int i = 0; i < NormalDist.size(); i++)
		{
			for (int j = 0; j < NormalDist[i].size(); j++)
			{
				auto& elem = NormalDist[i][j];
				if (std::get<2>(elem) == mat.rows * mat.columns)
				{
					mat(i, j) = FLOOR::UNKNOWN;
				}
				else if (phi((THRESHOLD - std::get<0>(elem)) / std::get<1>(elem)) > 0.6)// Normal Dist check
				{
					if (std::get<2>(elem) > static_cast<int>(mat.rows * mat.columns / 10))//more than 10% unmapped
						mat(i, j) = FLOOR::UNKNOWN;
					else
						mat(i, j) = FLOOR::WALKABLE;
				}
				else
					mat(i, j) = FLOOR::OBSTACLE;
			}
		}

		roomRenderer::msInfo = NormalDist;
	}

	void initSegment(ENTRANCE_DIR entrance, std::vector<roomSegment>& segments)
	{
		appendSegment(segments, roomSegment(sl::float2(0, 7), sl::float2(0, 8), roomSegment::SegmentType::DOOR));
		appendSegment(segments, roomSegment(sl::float2(0, 8), sl::float2(0, 9), roomSegment::SegmentType::DOOR));
	}

	void floodFillInternal(sl::float2 coords, std::vector<sl::float2>& stack, std::vector<sl::float2>& visited, MatClass<FLOOR>& mat, MatClass<int>& distanceMap)
	{
		visited.emplace_back(coords);
		distanceMap(coords.x, coords.y) = 1;
		if (coords.x > 0)
		{
			sl::float2 neighbour = sl::float2(coords.x - 1, coords.y);
			if (mat(neighbour.x, neighbour.y) == FLOOR::WALKABLE &&
				//std::find_if(visited.begin(), visited.end(), [&neighbour](const sl::float2& a) {return neighbour == a; }) == visited.end())
				distanceMap(neighbour.x, neighbour.y) != 1)
				//stack.emplace_back(neighbour);
				floodFillInternal(neighbour, stack, visited, mat, distanceMap);
		}
		if (coords.x < mat.rows - 1) {
			sl::float2 neighbour = sl::float2(coords.x + 1, coords.y);
			if (mat(neighbour.x, neighbour.y) == FLOOR::WALKABLE &&
				//std::find_if(visited.begin(), visited.end(), [&neighbour](const sl::float2& a) {return neighbour == a; }) == visited.end())
				distanceMap(neighbour.x, neighbour.y) != 1)
				//stack.emplace_back(neighbour);
				floodFillInternal(neighbour, stack, visited, mat, distanceMap);
		}
		if (coords.y > 0) {
			sl::float2 neighbour = sl::float2(coords.x, coords.y - 1);
			if (mat(neighbour.x, neighbour.y) == FLOOR::WALKABLE &&
				//std::find_if(visited.begin(), visited.end(), [&neighbour](const sl::float2& a) {return neighbour == a; }) == visited.end())
				distanceMap(neighbour.x, neighbour.y) != 1)
				//stack.emplace_back(neighbour);
				floodFillInternal(neighbour, stack, visited, mat, distanceMap);
		}
		if (coords.y < mat.columns - 1) {
			sl::float2 neighbour = sl::float2(coords.x, coords.y + 1);
			if (mat(neighbour.x, neighbour.y) == FLOOR::WALKABLE &&
				//std::find_if(visited.begin(), visited.end(), [&neighbour](const sl::float2& a) {return neighbour == a; }) == visited.end())
				distanceMap(neighbour.x, neighbour.y) != 1)
				//stack.emplace_back(neighbour);
				floodFillInternal(neighbour, stack, visited, mat, distanceMap);
		}
		return;
		//if (stack.empty())
		//	return;
		//auto last = stack.back();
		//stack.pop_back();
		//floodFillInternal(last, stack, visited, mat, distanceMap);
	}

	void distanceMapping(MatClass<int>& distanceMap)
	{
		const sl::float2 doorLocation = sl::float2(0, 7);
		for (size_t i = 0; i < distanceMap.rows; i++)
			for (size_t j = 0; j < distanceMap.columns; j++) {
				if (distanceMap(i, j) != -1)
					distanceMap(i, j) = std::min(abs(i - doorLocation.x) + abs(j - doorLocation.y), abs(i - doorLocation.x) + abs(j - doorLocation.y - 1));
			}
	}

	// For any walkable piece of space in room that is in edge of a map run a scan
	//return true and set that element to portal if its in the room's corner or neighbours unmapped space
	bool checkNeighboursInternal(MatClass<FLOOR>& mat, int i, int j)
	{
		bool flag = false;

		for( int x = i - 1; x < i + 2; x++)
		{
			if (x < 0 || x >= mat.rows)
				continue;
			int y = x == i ? j - 1 : j;
			int uppery = x == i ? j + 2 : j + 1;
			for (y; y < uppery; y++)
			{
				if (y < 0 || y >= mat.columns || (x == j && y == i))
					continue;
				if (mat(x, y) == FLOOR::UNKNOWN)
				{
					flag = true;
				}
			}
		}
		if ((i == 0 && j == 0) || (i == 0 && j == mat.columns - 1) || (i == mat.rows - 1 && j == 0) || (i == mat.rows - 1 && j == mat.columns - 1))
		{
			mat(i, j) = FLOOR::PORTAL;
			return true;
		}
		return flag;
	}

	void findCorridorToRoomEdges(MatClass<FLOOR>& mat, std::vector<roomSegment>& segments)
	{
		auto condition = [&segments, &mat](int i, int j)
		{
			for( int x = i - 1; x < i + 2; x++)
			{

				if (x < 0 || x >= mat.rows)
				{
					continue;
				}
				unsigned int y = x == i ? j - 1 : j;
				unsigned int uppery = x == i ? j + 2 : j + 1;

				for (y; y < uppery; y++)
				{

					if (y < 0 || y >= mat.columns || (x == j && y == i)) {
						continue;
					}
					else if (mat(x, y) != FLOOR::ROOM)
					{
						if (x < i) {
							appendSegment(segments, roomSegment(sl::float2(i, j), sl::float2(i, j + 1), mat(x, y)));
						}
						else if (x == i && y < j) {
							appendSegment(segments, roomSegment(sl::float2(i, j), sl::float2(i + 1, j), mat(x, y)));
						}
						else if (x == i && y > j) {
							appendSegment(segments, roomSegment(sl::float2(i, j + 1), sl::float2(i + 1, j + 1), mat(x, y)));
						}
						else if (x > i) {
							appendSegment(segments, roomSegment(sl::float2(i + 1, j), sl::float2(i + 1, j + 1), mat(x, y)));
						}
					}
				}
			}
		};

		for( size_t i = 0; i < mat.rows; i++)
		{
			for( size_t j = 0; j < mat.columns; j++)
			{
				auto& pixel = mat(i, j);
				if (pixel == FLOOR::ROOM) {
					condition(i, j);
				}
			}
		}
		for (size_t i = 0; i < mat.columns; i++) 
		{
			if (mat(0, i) == FLOOR::ROOM)
				appendSegment(segments, roomSegment(sl::float2(0, i), sl::float2(0, i + 1), FLOOR::PORTAL));
			if (mat(mat.rows-1, i) == FLOOR::ROOM)
				appendSegment(segments, roomSegment(sl::float2(mat.rows -1, i), sl::float2(mat.rows - 1, i + 1), FLOOR::PORTAL));
			if (mat(i, 0) == FLOOR::ROOM)
				appendSegment(segments, roomSegment(sl::float2(i, 0), sl::float2(i + 1, 0), FLOOR::PORTAL));
			if (mat(i, mat.columns - 1) == FLOOR::ROOM)
				appendSegment(segments, roomSegment(sl::float2(i, mat.columns), sl::float2(i + 1, mat.columns), FLOOR::PORTAL));
		}
		return;
	}

	void floodFillSpaceInternal(MatClass<FLOOR>& mat, pixelBlock & block, std::vector<std::unique_ptr<pixelBlock>>& stack)
	{
		block.setVisited(mat);
		block.expand(mat, stack);
		if (stack.empty())
			return;
		auto last = std::move(stack.back());
		stack.pop_back();
		floodFillSpaceInternal(mat, *last, stack);
	}

	void floodFillSpace_Init(MatClass<FLOOR>& mat)
	{
		std::vector<std::unique_ptr<pixelBlock>> stack;
		pixelBlock_4 initBlock(sl::float2(0, 7), sl::float2(0, 8), sl::float2(1, 7), sl::float2(1, 8));
		initBlock.setVisited(mat);
		initBlock.expand(mat, stack);
		if (stack.empty())
			return;
		auto last = std::move(stack.back());
		stack.pop_back();
		floodFillSpaceInternal(mat, *last, stack);
	}

	void floodFillSpace(MatClass<FLOOR>& mat)
	{
		std::vector<std::unique_ptr<pixelBlock>> stack;
		pixelBlock_4 initBlock(sl::float2(0, 7), sl::float2(0, 8), sl::float2(1, 7), sl::float2(1, 8));
		if (mat(0, 7) != FLOOR::WALKABLE || mat(0, 8) != FLOOR::WALKABLE || mat(1, 7) != FLOOR::WALKABLE || mat(1, 8) != FLOOR::WALKABLE)
			return;
		initBlock.setVisited(mat);
		initBlock.expand(mat, stack);
		if (stack.empty())
			return;
		auto last = std::move(stack.back());
		stack.pop_back();
		floodFillSpaceInternal(mat, *last, stack);
	}

	pixelBlock_2 findEntranceToRoom(MatClass<FLOOR>& mat, MatClass<int>& distanceMap, rectangleID & rectangle)
	{
		sl::float2 minimum, minimum2;
		float min = mat.rows + mat.columns, min2 = min;
		for( size_t i = rectangle.left; i < rectangle.right; i++)
			for( size_t j = rectangle.bottom; j < rectangle.top; j++) {
				if (distanceMap(j, i) < min) {
					minimum2 = minimum;
					min2 = min;
					minimum.x = j;
					minimum.y = i;
					min = distanceMap(j, i);
				}
				else if (distanceMap(j, i) >= min && distanceMap(j, i) < min2) {
					minimum2.x = j;
					minimum2.y = i;
					min2 = distanceMap(j, i);
				}
			}
		return pixelBlock_2(minimum, minimum2);
	};

	void findCorridorToRoom(MatClass<FLOOR>& mat, MatClass<int>& distanceMap, rectangleID & rectangle)
	{
		auto startPoint = findEntranceToRoom(mat, distanceMap, rectangle);
		std::vector<pixelBlock_4> stack;

		if (startPoint.checkIfEntranceReached(distanceMap))
		{
			//present result on debug
			presentresult(mat, rectangle);
			return;
		}

		auto startBlock = startPoint.findClosest(mat, distanceMap);
		bool flag = startPoint.expansionFlag;
		stack.emplace_back(startBlock);

		while (flag && !stack.back().checkIfEntranceReached(distanceMap))
		{
			auto newBlock = stack.back().findClosest(mat, distanceMap);
			stack.emplace_back(newBlock);
			flag = stack.back().expansionFlag;
		}

		if (flag)
		{
			for (auto& s : stack)
				s.setVisited(mat);
		}
		else
		{
			floodFillSpace(mat);
		}
#if DEBUG
		print(mat);
#endif // DEBUG

	}

	void alignToWorldAxis_Internal(std::vector<roomSegment>& segments, ENTRANCE_DIR entrance, int matsize)
	{
		auto iterator = [&segments = segments](std::function<void(roomSegment&)> functor) {
			for (auto& seg : segments)
				functor(seg);
		};

		switch (entrance)
		{
		case ENTRANCE_DIR::X_DOWN:
			iterator([matsize](roomSegment& seg) 
			{
				auto temp = seg;
				seg.pointA.x = -temp.pointA.x + matsize;
				seg.pointA.y = -temp.pointA.y + matsize;
				seg.pointB.x = -temp.pointB.x + matsize;
				seg.pointB.y = -temp.pointB.y + matsize;
				seg.pointA -= sl::float2(16, 8);
				seg.pointB -= sl::float2(16, 8);
				switch (seg.direction)
				{
				case GM::ENTRANCE_DIR::WALL:
					break;
				case GM::ENTRANCE_DIR::X_DOWN:
					seg.direction = GM::ENTRANCE_DIR::X_UP;
					break;
				case GM::ENTRANCE_DIR::X_UP:
					seg.direction = GM::ENTRANCE_DIR::X_DOWN;
					break;
				case GM::ENTRANCE_DIR::Y_DOWN:
					seg.direction = GM::ENTRANCE_DIR::Y_UP;
					break;
				case GM::ENTRANCE_DIR::Y_UP:
					seg.direction = GM::ENTRANCE_DIR::Y_DOWN;
					break;
				default:
					break;
				}
			});
			break;
		case ENTRANCE_DIR::Y_UP:
			iterator([matsize](roomSegment& seg) 
			{
				auto temp = seg;
				seg.pointA.x = -temp.pointA.y + matsize;
				seg.pointA.y = temp.pointA.x;
				seg.pointB.x = -temp.pointB.y + matsize;
				seg.pointB.y = temp.pointB.x;
				seg.pointA -= sl::float2(8, 0);
				seg.pointB -= sl::float2(8, 0);
				switch (seg.direction)
				{
				case GM::ENTRANCE_DIR::WALL:
					break;
				case GM::ENTRANCE_DIR::X_DOWN:
					seg.direction = GM::ENTRANCE_DIR::Y_DOWN;
					break;
				case GM::ENTRANCE_DIR::X_UP:
					seg.direction = GM::ENTRANCE_DIR::Y_UP;
					break;
				case GM::ENTRANCE_DIR::Y_DOWN:
					seg.direction = GM::ENTRANCE_DIR::X_UP;
					break;
				case GM::ENTRANCE_DIR::Y_UP:
					seg.direction = GM::ENTRANCE_DIR::X_DOWN;
					break;
				default:
					break;
				}
			});
			break;
		case ENTRANCE_DIR::Y_DOWN:
			iterator([matsize](roomSegment& seg) 
			{
				auto temp = seg;
				seg.pointA.x = temp.pointA.y;
				seg.pointA.y = matsize - temp.pointA.x;
				seg.pointB.x = temp.pointB.y;
				seg.pointB.y = matsize - temp.pointB.x;
				seg.pointA -= sl::float2(8, 16);
				seg.pointB -= sl::float2(8, 16);
				switch (seg.direction)
				{
				case GM::ENTRANCE_DIR::WALL:
					break;
				case GM::ENTRANCE_DIR::X_DOWN:
					seg.direction = GM::ENTRANCE_DIR::Y_UP;
					break;
				case GM::ENTRANCE_DIR::X_UP:
					seg.direction = GM::ENTRANCE_DIR::Y_DOWN;
					break;
				case GM::ENTRANCE_DIR::Y_DOWN:
					seg.direction = GM::ENTRANCE_DIR::X_DOWN;
					break;
				case GM::ENTRANCE_DIR::Y_UP:
					seg.direction = GM::ENTRANCE_DIR::X_UP;
					break;
				default:
					break;
				}
			});
			break;
		default:
			iterator([matsize](roomSegment& seg)
			{
				seg.pointA -= sl::float2(0, 8);
				seg.pointB -= sl::float2(0, 8);
			});
			break;
		}
	}

	void alignToWorldAxisInit_Internal(std::vector<roomSegment>& segments, sl::float2 location)
	{
		for (auto& segment : segments)
		{
			segment.pointA -= sl::float2(1, 8);
			segment.pointB -= sl::float2(1, 8);
		}
		
	}

	void alignToWorldAxis_Internal(std::vector<GlobalMappingInformation>& segments, ENTRANCE_DIR entrance, int matsize)
	{
		auto iterator = [&segments = segments](std::function<void(GlobalMappingInformation&)> functor)
		{
			for (auto& seg : segments)
				functor(seg);
		};

		switch (entrance)
		{
		case ENTRANCE_DIR::X_DOWN:
			iterator([matsize](GlobalMappingInformation& seg)
			{
				seg.position.x = -seg.position.x + matsize;
				seg.position.y = -seg.position.y + matsize;
				seg.position -= sl::float2(16, 8);
			});
			break;
		case ENTRANCE_DIR::Y_UP:
			iterator([matsize](GlobalMappingInformation& seg)
			{
				auto temp = seg;
				seg.position.x = -temp.position.y + matsize;
				seg.position.y = temp.position.x;
				seg.position -= sl::float2(8, 0);
			});
			break;
		case ENTRANCE_DIR::Y_DOWN:
			iterator([matsize](GlobalMappingInformation& seg)
			{
				auto temp = seg;
				seg.position.x = temp.position.y;
				seg.position.y = matsize - temp.position.x;
				seg.position -= sl::float2(8, 16);
			});
			break;
		default:
			iterator([matsize](GlobalMappingInformation& seg)
				{
					auto temp = seg;
					seg.position -= sl::float2(0, 8);
				});
			break;
		}
	}

	void alignToWorldAxisInit_Internal(std::vector<GlobalMappingInformation>& segments)
	{
		for (auto& segment : segments)
		{
			segment.position -= sl::float2(1, 8);
		}

	}

	void iterator(MatClass<FLOOR>& mat, std::vector<GlobalMappingInformation>& vec, std::function<void(std::vector<GlobalMappingInformation>&, MatClass<FLOOR>&, int, int)> functor)
	{
		for (unsigned int i = 0; i < mat.rows; i++)
			for (unsigned int j = 0; j < mat.columns; j++)
				if (mat(i, j) == FLOOR::ROOM)
					functor(vec, mat, i, j);
	}
}

roomRenderer::roomRenderer(sl::float2 location, ENTRANCE_DIR in_entrance)
: matsize(16), entrance(in_entrance), location(location), mat(matsize, matsize, FLOOR::DOWNSAMPLED_INIT)
{
	initSegment(entrance, segments);
	int ilow, jlow;

	//For release version
	std::unique_lock<std::mutex> lock(GlobalMapDevice::mutex());
	MatClass<float> tempMat(256, 256, GlobalMapDevice::copyGlobalMap(location, entrance, ilow, jlow));
	lock.unlock();
	//For debug version
	//MatClass<FLOOR> tempMat(256, 256, FLOOR::UNKNOWN, MatClassBase::GPUOPTIMIZATION::DISABLED);
	//copyGlobalMap(location, entrance, tempMat, ilow, jlow);

	erode(tempMat);
	downsample(tempMat, mat);

	DebugRender(tempMat);
	DebugRender();
	
	for (int i = 0; i < matsize; i++)
	{
		dInfo.emplace_back(std::vector<std::pair<FLOOR, std::vector<std::vector<float>>>>(16, std::make_pair(FLOOR::UNKNOWN, std::vector<std::vector<float>>(16, std::vector<float>(16, -1)))));
		for (int j = 0; j < matsize; j++) {
			dInfo[i][j].first = mat(i, j);

			for (int r = i * downsampleFactor; r < i * downsampleFactor + downsampleFactor; r++)
			{
				for (int c = j * downsampleFactor; c < j * downsampleFactor + downsampleFactor; c++)
				{
					dInfo[i][j].second[r % downsampleFactor][c % downsampleFactor] = tempMat(r, c);
				}
			}

		}
	}

#if DEBUG
	print(mat);
#endif // DEBUG

	MatClass<int> distanceMap(matsize, matsize, -1);
	floodFill(segments[0].pointA, distanceMap);
	distanceMapping(distanceMap);

#if DEBUG
	print(distanceMap);
#endif // DEBUG

	findPortals(distanceMap, &checkNeighbours);
	auto room = orlowski(mat).maxRectangle;
	//room.isBigEnough() ? findCorridorToRoom(mat, distanceMap, room) : floodFillSpace(mat);
	//presentresult(mat, room);//room is ready
	if (room.isBigEnough())
	{
		findCorridorToRoom(mat, distanceMap, room);
		presentresult(mat, room);
	}
	else
	floodFillSpace(mat);
	corridorGenerator();

	////update the room vector
	//int ilow = 0;
	//int jlow = 0;

#if DEBUG
	print(mat);
#endif // DEBUG

	getRoomCells(ilow, jlow);
	alignToWorldAxis();
}

void mouse_callback(int  event, int  x, int  y, int  flag, void* param)
{
	if (event == cv::EVENT_MOUSEMOVE)
	{
		std::cout << "(" << x << ", " << y << ")" << std::endl;
	}
}

roomRenderer::roomRenderer(sl::float2 location)
	: matsize(16), entrance(ENTRANCE_DIR::X_UP), location(location), mat(matsize, matsize, FLOOR::DOWNSAMPLED_INIT)
{
	int ilow, jlow;

	//For release version
	std::unique_lock<std::mutex> lock (GlobalMapDevice::mutex ());
	MatClass<float> tempMat(256, 256, GlobalMapDevice::copyGlobalMap(location, ilow, jlow));
	lock.unlock();
	//For debug version
	//MatClass<FLOOR> tempMat(256, 256, FLOOR::UNKNOWN, MatClassBase::GPUOPTIMIZATION::DISABLED);
	//copyGlobalMapInit(location, tempMat, ilow, jlow);

	erode(tempMat);
	downsample(tempMat, mat);
	//Cheat
	mat(0, 7) = mat(0, 8) = mat(1, 7) = mat(1, 8) = FLOOR::WALKABLE;
	mat(2, 7) = mat(2, 8) = mat(3, 7) = mat(3, 8) = FLOOR::WALKABLE;
	//Debug
	//DebugRender(tempMat);
	//DebugRender();

	for (int i = 0; i < matsize; i++)
	{
		dInfo.emplace_back(std::vector<std::pair<FLOOR, std::vector<std::vector<float>>>>(16, std::make_pair(FLOOR::UNKNOWN, std::vector<std::vector<float>>(16, std::vector<float>(16, -1)))));
		for (int j = 0; j < matsize; j++) {
			dInfo[i][j].first = mat(i, j);

			for (int r = i * downsampleFactor; r < i * downsampleFactor + downsampleFactor; r++)
			{
				for (int c = j * downsampleFactor; c < j * downsampleFactor + downsampleFactor; c++)
				{
					dInfo[i][j].second[r % downsampleFactor][c % downsampleFactor] = tempMat(r, c);
				}
			}
			
		}
	}
	//End Debug

#if DEBUG
	print(mat);
#endif // DEBUG

	MatClass<int> distanceMap(matsize, matsize, -1);
	floodFill(sl::float2(1, 7), distanceMap);

	const sl::float2 doorLocation = sl::float2(1, 7);
	for (size_t i = 0; i < distanceMap.rows; i++)
	{
		for (size_t j = 0; j < distanceMap.columns; j++)
		{
			if (distanceMap(i, j) != -1)
				distanceMap(i, j) = std::min(abs(i - doorLocation.x) + abs(j - doorLocation.y), abs(i - doorLocation.x) + abs(j - doorLocation.y - 1));
		}
	}

#if DEBUG
	print(distanceMap);
#endif // DEBUG

	findPortals(distanceMap, &checkNeighboursInit);
	floodFillSpace_Init(mat);
	corridorGenerator(true);

#if DEBUG
	print(mat);
#endif // DEBUG

	getRoomCells(ilow, jlow);
	alignToWorldAxisInit();
}

void roomRenderer::floodFill(sl::float2 coords, MatClass<int>& distanceMap)
{
	std::vector<sl::float2> stack, visited;
	floodFillInternal(coords, stack, visited, mat, distanceMap);
}

void roomRenderer::findPortals(MatClass<int>& distanceMap, const std::function<bool(roomRenderer*, int, int)>& func)
{

	for( size_t i = 0; i < mat.rows; i++) {
		for( size_t j = 0; j < mat.columns; j++) {
			if (mat(i, j) == FLOOR::WALKABLE) {

				if (distanceMap(i, j) == -1) {
					mat(i, j) = FLOOR::UNKNOWN;
					continue;
				}
				//checkNeighbours(i, c);
				//func(this, i, j);
			}

		}
	}

}

void roomRenderer::alignToWorldAxis()//to class
{
	decideSegmentsOrientation();
	alignToWorldAxis_Internal(segments, entrance, matsize);
	alignToWorldAxis_Internal(roomCells, entrance, matsize);

	for (auto& segment : segments) {
		segment.pointA = location + sl::float2(64 * segment.pointA.x, 64 * segment.pointA.y);
		segment.pointB = location + sl::float2(64 * segment.pointB.x, 64 * segment.pointB.y);
	}

	for (auto& roomCell : roomCells)
	{
		roomCell.position = location + sl::float2(64 * roomCell.position.x, 64 * roomCell.position.y);
	}
}

void roomRenderer::alignToWorldAxisInit()//to class
{
	decideSegmentsOrientation();
	alignToWorldAxisInit_Internal(segments, matsize);
	alignToWorldAxisInit_Internal(roomCells);

	for (auto& segment : segments)
	{
		segment.pointA = location + sl::float2( 64 * segment.pointA.x, 64 * segment.pointA.y);
		segment.pointB = location + sl::float2( 64 * segment.pointB.x, 64 * segment.pointB.y);
	}

	for (auto& roomCell : roomCells)
	{
		roomCell.position = location + sl::float2(64 * roomCell.position.x, 64 * roomCell.position.y);
	}
}

bool roomRenderer::checkNeighbours(roomRenderer* ptr, int i, int j)
{
	//call condition lambda for any edge (walkable) element
	//check if edge case is i==0 and if element is not corner element or neighbouring unknown element
	//then add segment to the list and keep it as walkable space otherwise set it to portal
	auto condition = [ptr](int i, int j) {
		if (i == 0 && !checkNeighboursInternal(ptr->mat, i, j))
			appendSegment(ptr->segments, roomSegment(sl::float2(i, j), sl::float2(i, j + 1)));
		else
			ptr->mat(i, j) = FLOOR::PORTAL;
	};

	//check each element's 4 neighbours
	for( int x = i - 1; x < i + 2; x++) {
		if (x < 0 || x >= ptr->mat.rows) {
			condition(i, j);
			continue;
		}

		int y = x == i ? j - 1 : j;
		int uppery = x == i ? j + 2 : j + 1;

		for (y; y < uppery; y++) 
		{
			if (y < 0 || y >= ptr->mat.columns)
			{
				condition(i, j);
				continue;
			}
			else if (x == j && y == i)
				continue;
			if (ptr->mat(x, y) == FLOOR::UNKNOWN)
			{
				ptr->mat(i, j) = FLOOR::PORTAL;
				return true;
			}
		}

	}

	return false;
}

bool roomRenderer::checkNeighboursInit(roomRenderer * ptr, int i, int j)
{
	//check each element's 4 neighbours
	for (int x = i - 1; x < i + 2; x++)
	{
		if (x < 0 || x >= ptr->mat.rows)
		{
			//condition(i, c);
			ptr->mat(i, j) = FLOOR::PORTAL;
			continue;
		}

		int y = x == i ? j - 1 : j;
		int uppery = x == i ? j + 2 : j + 1;

		for (y; y < uppery; y++)
		{
			if (y < 0 || y >= ptr->mat.columns)
			{
				//condition(i, c);
				ptr->mat(i, j) = FLOOR::PORTAL;
				continue;
			}
			else if (x == j && y == i)
				continue;
			if (ptr->mat(x, y) == FLOOR::UNKNOWN)
			{
				ptr->mat(i, j) = FLOOR::PORTAL;
				return true;
			}
		}

	}

	return false;
}

void GM::roomRenderer::DebugRender(MatClass<float>& tempMat)
{
	cv::Mat image(256, 256, CV_32FC3);
	std::vector<std::vector<float>> tm;
	for (int i = 0; i < 256; i++)
	{
		tm.emplace_back(std::vector<float>());
		for (int j = 0; j < 256; j++)
		{
			tm.back().emplace_back(tempMat(i, j));
			auto color = cv::Vec3f(0, 0, 0);
			if (tempMat(i, j) > 0 && tempMat(i, j) < THRESHOLD)
				color = cv::Vec3f(1, 1, 1);
			else if (tempMat(i, j) >= THRESHOLD)
				color = cv::Vec3f(0, 0, 255);
			image.at<cv::Vec3f>(255 - i, j) = color;
		}
	}
	cv::resize(image, image, cv::Size(512, 512));

	for (int i = 32; i < 512; i += 32)
		cv::line(image, cv::Point(0, i), cv::Point(512, i), cv::Scalar(0, 255, 0), 1);

	for (int i = 32; i < 512; i += 32)
		cv::line(image, cv::Point(i, 0), cv::Point(i, 512), cv::Scalar(0, 255, 0), 1);

	cv::drawMarker(image, cv::Point(256, 256), cv::Scalar(255, 0, 0));
	cv::namedWindow("window");
	cv::imshow("window", image);
	cv::waitKey(0);
}

void GM::roomRenderer::DebugRender()
{
	cv::Mat image(256, 256, CV_32FC3);
	for (int i = 0; i < 16; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			auto color = cv::Vec3f(0, 0, 0);
			if (mat(i, j) == FLOOR::WALKABLE)
				color = cv::Vec3f(1, 1, 1);
			else if (mat(i, j) == FLOOR::OBSTACLE)
				color = cv::Vec3f(0, 0, 255);
			for (int row = 0; row < 16; row++)
			{
				for (int col = 0; col < 16; col++)
				{
					image.at<cv::Vec3f>(255 - (i * 16 +  row), j * 16 + col) = color;
				}
			}
		}
	}
	cv::resize(image, image, cv::Size(512, 512));

	for (int i = 32; i < 512; i += 32)
		cv::line(image, cv::Point(0, i), cv::Point(512, i), cv::Scalar(0, 255, 0), 1);

	for (int i = 32; i < 512; i += 32)
		cv::line(image, cv::Point(i, 0), cv::Point(i, 512), cv::Scalar(0, 255, 0), 1);

	cv::drawMarker(image, cv::Point(256, 256), cv::Scalar(255, 0, 0));
	cv::namedWindow("window");
	cv::imshow("window", image);
	cv::waitKey(0);
}

void roomRenderer::corridorGenerator(bool onLoad)
{

	findCorridorToRoomEdges(mat, segments);

	if (segments.size() == 0 && !onLoad) return;

	sortSegment(this->segments);

	if (segments.size() >= 2 && segments[0].segmentType == roomSegment::SegmentType::DOOR &&
		segments[1].segmentType == roomSegment::SegmentType::DOOR && !onLoad)
	{
	segments.erase(segments.begin(), segments.begin() + 2);
	}

	if (!onLoad && segments.size() > 0) {
		segments.front().segmentType = roomSegment::SegmentType::WALL;
		segments.back().segmentType = roomSegment::SegmentType::WALL;
	}

	decidePortals(this->segments);
}

void roomRenderer::decideSegmentsOrientation()//to class
{
	for (auto& segment : segments) 
	{
		if (segment.segmentType != roomSegment::SegmentType::WALL) 
		{
			segment.segmentType = roomSegment::SegmentType::DOOR;
		}

		if (segment.pointA.x == segment.pointB.x) 
		{

			if ((segment.pointA.x != 0) && (mat(segment.pointA.x - 1, std::min(segment.pointA.y, segment.pointB.y)) == FLOOR::ROOM))
				segment.direction = ENTRANCE_DIR::X_UP;
			else
				segment.direction = ENTRANCE_DIR::X_DOWN;
		}

		else if (segment.pointA.y == segment.pointB.y) 
		{
			if (segment.pointA.y != 0 && mat(std::min(segment.pointA.x, segment.pointB.x), segment.pointA.y - 1) == FLOOR::ROOM)
				segment.direction = ENTRANCE_DIR::Y_UP;
			else
				segment.direction = ENTRANCE_DIR::Y_DOWN;
		}

	}
}

void roomRenderer::getRoomCells(int i_low, int j_low)
{
	for (unsigned int i = 0; i < mat.rows; i++)
		for (unsigned int j = 0; j < mat.columns; j++)
			if (mat(i, j) == FLOOR::ROOM)
			{
				GlobalMappingInformation info;
				info.position = sl::float2(i, j);
				roomCells.emplace_back(info);
			}
}
