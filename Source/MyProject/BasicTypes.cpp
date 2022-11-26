// Fill out your copyright notice in the Description page of Project Settings.

#include "BasicTypes.h"


FRenderingInformation::FRenderingInformation(FVector2D PointA, FVector2D PointB, bool IsWall)
	: PointA(PointA), PointB(PointB), IsWall(IsWall)
{}

FRenderingInformation::FRenderingInformation(const GM::roomSegment & other)
{
	PointA = sl::unreal::ToUnrealType(other.pointA);
	PointB = sl::unreal::ToUnrealType(other.pointB);
	DirectionInt = Utils::ToEntranceWrapperClass(other.direction);
	//Direction.operator=( ToEntranceWrapper(other.typeOfSegment));
	IsWall = (other.segmentType == GM::roomSegment::SegmentType::WALL);
}

FGlobalMappingInformation::FGlobalMappingInformation(int32 ILow, int32 JLow, int32 IHigh, int32 JHigh)
	: ILow(ILow), JLow(JLow), IHigh(IHigh), JHigh(JHigh), Counter(0)
{}

namespace Utils 
{
	GM::ENTRANCE_DIR StripEntranceWrapper(TEnumAsByte<EntranceDirection> Entrance)
{
	switch (Entrance)
	{
	case MINUS_X:
		return GM::ENTRANCE_DIR::X_DOWN;
	case PLUS_X:
		return GM::ENTRANCE_DIR::X_UP;
	case MINUS_Y:
		return GM::ENTRANCE_DIR::Y_DOWN;
	case PLUS_Y:
		return GM::ENTRANCE_DIR::Y_UP;
	default:
		return GM::ENTRANCE_DIR::WALL;
	}
}

EntranceDirection ToEntranceWrapper(GM::ENTRANCE_DIR Entrance)
{
	switch (Entrance)
	{
	case GM::ENTRANCE_DIR::X_UP:
		return EntranceDirection::PLUS_X;
	case GM::ENTRANCE_DIR::X_DOWN:
		return EntranceDirection::MINUS_X;
	case GM::ENTRANCE_DIR::Y_DOWN:
		return EntranceDirection::MINUS_Y;
	case GM::ENTRANCE_DIR::Y_UP:
		return EntranceDirection::PLUS_Y;
	default:
		return EntranceDirection::WALL;
	}
}

GM::ENTRANCE_DIR StripEntranceWrapper(EntranceDirectionClass Entrance)
{
	switch (Entrance)
	{
	case EntranceDirectionClass::MINUS_X:
		return GM::ENTRANCE_DIR::X_DOWN;
	case EntranceDirectionClass::PLUS_X:
		return GM::ENTRANCE_DIR::X_UP;
	case EntranceDirectionClass::MINUS_Y:
		return GM::ENTRANCE_DIR::Y_DOWN;
	case EntranceDirectionClass::PLUS_Y:
		return GM::ENTRANCE_DIR::Y_UP;
	default:
		return GM::ENTRANCE_DIR::WALL;
	}
}

EntranceDirectionClass ToEntranceWrapperClass(GM::ENTRANCE_DIR Entrance)
{
	switch (Entrance)
	{
	case GM::ENTRANCE_DIR::X_UP:
		return EntranceDirectionClass::PLUS_X;
	case GM::ENTRANCE_DIR::X_DOWN:
		return EntranceDirectionClass::MINUS_X;
	case GM::ENTRANCE_DIR::Y_DOWN:
		return EntranceDirectionClass::MINUS_Y;
	case GM::ENTRANCE_DIR::Y_UP:
		return EntranceDirectionClass::PLUS_Y;
	default:
		return EntranceDirectionClass::WALL;
	}
}
}