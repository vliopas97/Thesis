// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "sl/Camera.hpp"
#include "Stereolabs/StereolabsCoreGlobals.h"
#include "roomSegment.h"
#include "BasicTypes.generated.h"

class AObstacleBase;
class AHumanBase;
class AHumanObstacleAI;

struct ObstacleInformation
{
	ObstacleInformation() = default;
	ObstacleInformation(const GM::GlobalMappingInformation& input) : GlobalMappingInformation(input) {}

	GM::GlobalMappingInformation GlobalMappingInformation;
	FVector2D Position;
	int CounterPrevFrame;
	AObstacleBase* ActorPtr = nullptr;
};

struct HumanInformation 
{
	enum class State 
	{
		PendingDestruction, Active, New
	};

	int TrackID = -1;
	AHumanObstacleAI* ActorPtr = nullptr;
	State TrackState = State::New;
	FVector2D Position;
};

UENUM(BlueprintType)
enum EntranceDirection
{
	WALL = -1,
	MINUS_X = 0,
	PLUS_X,
	MINUS_Y,
	PLUS_Y
};

UENUM()
enum class EntranceDirectionClass : int32
{
	WALL = -1,
	MINUS_X = 0,
	PLUS_X,
	MINUS_Y,
	PLUS_Y
};

USTRUCT(BlueprintType, Category = "Stereolabs|Global Map")
struct MYPROJECT_API FRenderingInformation
{
	GENERATED_USTRUCT_BODY()

	FRenderingInformation(FVector2D PointA = FVector2D::ZeroVector, FVector2D PointB = FVector2D::ZeroVector, bool IsWall = true);

	FRenderingInformation(const GM::roomSegment& other);

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FVector2D PointA;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FVector2D PointB;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	TEnumAsByte<EntranceDirection> Direction;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	EntranceDirectionClass DirectionInt;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	bool IsWall;

};

USTRUCT(BlueprintType, Category = "Stereolabs|Global Map")
struct MYPROJECT_API FGlobalMappingInformation
{
	GENERATED_USTRUCT_BODY()

		FGlobalMappingInformation(int32 ILow = 0, int32 JLow = 0, int32 IHigh = 0, int32 JHigh = 0);

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		int32 ILow;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		int32 JLow;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		int32 IHigh;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		int32 JHigh;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		int32 Counter;

};

namespace Utils 
{

	GM::ENTRANCE_DIR StripEntranceWrapper(TEnumAsByte<EntranceDirection> Entrance);

	EntranceDirection ToEntranceWrapper(GM::ENTRANCE_DIR Entrance);

	GM::ENTRANCE_DIR StripEntranceWrapper(EntranceDirectionClass Entrance);

	EntranceDirectionClass ToEntranceWrapperClass(GM::ENTRANCE_DIR Entrance);
}