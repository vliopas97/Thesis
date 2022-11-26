#include "Orlowski.h"

using namespace GM;



namespace 
{
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
		for (int i = mat.rows - 1; i >= 0; i--) {
			std::cout << " ";
			for (size_t j = 0; j < mat.columns; j++)
				std::cout << mat(i, j) << " ";
			std::cout << "\n";
		}
		puts("");
	}

	//using rectangleID = orlowski::rectangleIDAlias;

	void presentresult(MatClass<FLOOR>& mat, const rectangleID& rectangle)
	{
		for (size_t i = rectangle.left; i < rectangle.right; i++)
			for (size_t j = rectangle.bottom; j < rectangle.top; j++)
			{
				mat(j, i) = FLOOR::ROOM;
			}
	}

	struct indexedFloat {

		indexedFloat(sl::float2& coords, int index)
			:
			coords(coords), index(index)
		{
		}

		indexedFloat(const indexedFloat& other) 
			:coords(other.coords)
		{
			this->index = other.index;
		}

		bool operator==(const indexedFloat& other) const
		{
			return this->coords == other.coords && this->index == other.index;
		}

		union 
		{
			sl::float2 coords;

			struct {
				float x, y;
			};
		};

		int index;
	};

	struct Tent {

		void appendLeft(const indexedFloat& element)
		{
			if (std::find_if(TL.begin(), TL.end(),
				[&element](const indexedFloat& a) { return a == element; }) != TL.end())
				return;
			TL.emplace(TL.begin(), element);
		}

		void appendRight(const indexedFloat& element)
		{
			if (std::find_if(TR.begin(), TR.end(),
				[&element](const indexedFloat& a) { return element == a; }) != TR.end())
				return;
			TR.emplace(TR.begin(), element);
		}

		std::vector<indexedFloat> TL, TR;//increasing order on y
	};	
	
	void initSets(MatClass<FLOOR>& mat, std::vector<sl::float2>& S, std::vector<sl::float2>& S1d)
	{
		for (size_t i = 0; i < mat.rows; i++)
			for (size_t j = 0; j < mat.columns; j++)
				if (mat(i, j) != FLOOR::WALKABLE) {
					S.emplace_back(sl::float2(j + 0.5, i + 0.5));
					S1d.emplace_back(sl::float2(j + 0.5, i + 0.5));
				}

			std::sort(S.begin(), S.end(), [](const sl::float2& a, const sl::float2& b) {
			return a.y < b.y;
		});

		S.emplace(S.begin(), sl::float2(mat.columns, 0));
		S.emplace_back(sl::float2(0, mat.rows));

		std::sort(S1d.begin(), S1d.end(), [](const sl::float2& a, const sl::float2& b) {
			return a.x < b.x;
		});

		S1d.emplace(S1d.begin(), sl::float2(0, mat.rows));
		S1d.emplace_back(sl::float2(mat.columns, 0));
	}

	void tent(const indexedFloat& PointI, const indexedFloat& PointJ, const indexedFloat& PointK, std::vector<Tent>& tents)
	{
		if (PointJ.index != 0) {
			tents[PointK.index].appendLeft(PointJ);
			tents[PointI.index].appendRight(PointJ);
		}

		PointI.y > PointK.y ? tents[PointK.index].appendLeft(PointI) : tents[PointI.index].appendRight(PointK);
	}

	rectangleID phaseI(std::vector<sl::float2>& S, std::vector<sl::float2>& S1d)
	{
		rectangleID Rbt, Rlr;
		for (size_t i = 0; i < S1d.size() - 1; i++) {
			Rbt.compare(rectangleID(S1d.back().y, S1d[i].x, S1d[0].y, S1d[i + 1].x));
		}

		for (size_t i = 0; i < S.size() - 1; i++) {
			Rlr.compare(rectangleID(S[i].y, S.back().x, S[i + 1].y, S[0].x));
		}

		return rectangleID::max(Rbt, Rlr);
	}

	rectangleID phaseII_bottom(std::vector<sl::float2>& S, std::vector<Tent>& tents)
	{
		rectangleID Rb;
		int i = 0, j = 1, k = 2;

		std::vector<indexedFloat> S1d;
		S1d.reserve(S.size());

		for (size_t it = 0; it < S.size(); it++) 
		{
			S1d.emplace_back(indexedFloat(S[it], it));
		}

		std::sort(S1d.begin(), S1d.end(), [](const indexedFloat& a, const indexedFloat& b) { return a.x < b.x; });

		while (!(i == 1 && k == S1d.size() - 1)) {

			if (S1d[j].y >= S1d[i].y && S1d[j].y >= S1d[k].y) 
			{
				Rb.compare(rectangleID(0, S1d[i].x, S1d[j].y, S1d[k].x));
				tent(S1d[i], S1d[j], S1d[k], tents);
				S1d.erase(S1d.begin() + j);
				j = i;
				i--;
				k--;
			}
			else 
			{
				i = j;
				j = k;
				k = std::min(int(S1d.size() - 1), k + 1);
				if (i == j && i == k) {
					i = 0;
					j = 1;
				}
			}
		}

		return Rb;
	}

	rectangleID phaseII_up(std::vector<sl::float2> S1d)
	{
		rectangleID Rb;
		int i = 0, j = 1, k = 2;

		std::reverse(S1d.begin(), S1d.end());

		while (!(i == 1 && k == S1d.size() - 1)) 
		{

			if (S1d[j].y <= S1d[i].y && S1d[j].y <= S1d[k].y) 
			{
				Rb.compare(rectangleID(S1d[j].y, S1d[j].x, S1d[0].y, S1d[i].x));
				S1d.erase(S1d.begin() + j);
				j = i;
				i--;
				k--;
			}
			else 
			{
				i = j;
				j = k;
				k = std::min(int(S1d.size() - 1), k + 1);
				if (i == j && i == k) {
					i = 0;
					j = 1;
				}
			}
		}

		return Rb;
	}

	rectangleID phaseII_left(std::vector<sl::float2> S)
	{
		rectangleID Rb;
		int i = 0, j = 1, k = 2;

		while (!(i == 1 && k == S.size() - 1)) 
		{

			if (S[j].x >= S[i].x && S[j].x >= S[k].x) 
			{
				Rb.compare(rectangleID(S[i].y, 0, S[k].y, S[j].x));
				S.erase(S.begin() + j);
				j = i;
				i--;
				k--;
			}
			else 
			{
				i = j;
				j = k;
				k = std::min(int(S.size() - 1), k + 1);
				if (i == j && i == k) {
					i = 0;
					j = 1;
				}
			}
		}

		return Rb;
	}

	rectangleID phaseII_right(std::vector<sl::float2> S)
	{
		rectangleID Rb;
		int i = 0, j = 1, k = 2;

		std::reverse(S.begin(), S.end());
		
		while (!(i == 1 && k == S.size() - 1)) {
			if (S[j].x <= S[i].x && S[j].x <= S[k].x) {
				Rb.compare(rectangleID(S[k].y, S[j].x, S[i].y, S[S.size() - 1].x));
				S.erase(S.begin() + j);
				j = i;
				i--;
				k--;
			}
			else {
				i = j;
				j = k;
				k = std::min(int(S.size() - 1), k + 1);
				if (i == j && i == k) {
					i = 0;
					j = 1;
				}
			}
		}
		return Rb;
	}

	rectangleID phaseII(std::vector<sl::float2>& S, std::vector<sl::float2>& S1d, std::vector<Tent>& tents)
	{
		auto Rb = phaseII_bottom(S, tents);
		auto Rl = phaseII_left(S);
		auto Rt = phaseII_up(S1d);
		auto Rr = phaseII_right(S);

		return rectangleID::max(rectangleID::max(Rb, Rt), rectangleID::max(Rl, Rr));
	}

	rectangleID phaseIII(std::vector<sl::float2>& S, std::vector<Tent>& tents)
	{
		rectangleID Rb;
		if (S.size() < static_cast<size_t>(4))
			return Rb;

		for (size_t i = 1; i < S.size() - static_cast<size_t>(4); i++) 
		{
			if (tents[i].TL.empty() || tents[i].TR.empty())
				continue;

			tent(tents[i].TL.back(), indexedFloat(S[0], 0), tents[i].TR.back(), tents);

			while (true) 
			{

				if (tents[i].TL.back().y > tents[i].TR.back().y) 
				{
					if (tents[i].TL.size() == 1)
						break;

					Rb.compare(rectangleID(S[i].y, tents[i].TL.rbegin()[1].coords.x, tents[i].TL.back().coords.y, tents[i].TR.back().coords.x));
					tent(tents[i].TL.rbegin()[1], tents[i].TL.back(), tents[i].TR.back(), tents);
					tents[i].TL.pop_back();
				}
				else if (tents[i].TL.back().y <= tents[i].TR.back().y) 
				{
					if (tents[i].TR.size() == 1)
						break;

					Rb.compare(rectangleID(S[i].y, tents[i].TL.back().coords.x, tents[i].TR.back().coords.y, tents[i].TR.rbegin()[1].coords.x));
					tent(tents[i].TL.back(), tents[i].TR.back(), tents[i].TR.rbegin()[1], tents);
					tents[i].TR.pop_back();
				}
			}
		}

		return Rb;
	}
}

orlowski::orlowski(GM::MatClass<FLOOR>& mat)
{
	std::vector<sl::float2> S;
	std::vector<sl::float2> S1d;
	rectangleID R1, R2, R3;
	initSets(mat, S, S1d);
	std::vector<Tent> tents(S.size(), Tent());
	R1 = phaseI(S, S1d);
	if (S.size() > 2) {
		R2 = phaseII(S, S1d, tents);
		R3 = phaseIII(S, tents);
	}

	maxRectangle = rectangleID::max(R1, rectangleID::max(R2, R3));

	//DEBUG CODE
	//presentresult(mat, maxRectangle);
#if DEBUG
	print(mat);
#endif // DEBUG

}