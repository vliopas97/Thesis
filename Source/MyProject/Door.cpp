// Fill out your copyright notice in the Description page of Project Settings.

#include "Door.h"
#include "Async/Async.h"
#include "roomRenderer.h"

// Sets default values
ADoor::ADoor()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

bool ADoor::GetLoadNewRoom()
{
	FScopeLock Lock(&LoadNewRoomSection);
	return LoadNewRoom;
}

void ADoor::SetLoadNewRoom(bool Input)
{
	FScopeLock Lock(&LoadNewRoomSection);
	LoadNewRoom = Input;
}

bool ADoor::GetEnableInteraction()
{
	FScopeLock Lock(&EnableInteractionSection);
	return EnableInteraction;
}

void ADoor::SetEnableInteraction(bool Input)
{
	FScopeLock Lock(&EnableInteractionSection);
	EnableInteraction = Input;
}

void ADoor::SetSegments(UPARAM(ref) TArray<FRenderingInformation>& Input)
{
	FScopeLock Lock(&SegmentsSection);
	Segments = MoveTemp(Input);
}

void ADoor::GetSegments(TArray<FRenderingInformation>& Output)
{
	FScopeLock Lock(&SegmentsSection); 
	Output = Segments;
}

void ADoor::DebuggerHelp()
{
	UE_LOG(LogTemp, Error, TEXT("New Room Actors size is: %d"), NewRoomActors.Num());
}

void ADoor::SwapDoorDirection()
{
	switch (Direction)
	{
	case EntranceDirectionClass::PLUS_X:
		Direction = EntranceDirectionClass::MINUS_X;
		break;
	case EntranceDirectionClass::MINUS_X:
		Direction = EntranceDirectionClass::PLUS_X;
		break;
	case EntranceDirectionClass::PLUS_Y:
		Direction = EntranceDirectionClass::MINUS_Y;
		break;
	case EntranceDirectionClass::MINUS_Y:
		Direction = EntranceDirectionClass::PLUS_Y;
		break;
	default:
		break;
	}
}

// Called when the game starts or when spawned
void ADoor::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void ADoor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}
