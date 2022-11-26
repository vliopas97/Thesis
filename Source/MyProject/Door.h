// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BasicTypes.h"
//#include "HAL/ThreadingBase.h"
#include "Door.generated.h"

enum EntranceDirection;

UCLASS()
class MYPROJECT_API ADoor : public AActor
{
	GENERATED_BODY()
	
public:

	UFUNCTION(BlueprintCallable)
	bool GetLoadNewRoom();

	UFUNCTION(BlueprintCallable)
	void SetLoadNewRoom(bool Input);

	UFUNCTION(BlueprintCallable)
	bool GetEnableInteraction();

	UFUNCTION(BlueprintCallable)
	void SetEnableInteraction(bool Input);

	UFUNCTION(BlueprintCallable)
	void SwapDoorDirection();

	UFUNCTION(BlueprintCallable)
	void SetSegments(UPARAM(ref) TArray<FRenderingInformation>& Input);

	UFUNCTION(BlueprintCallable)
	void GetSegments(TArray<FRenderingInformation>& Output);

	UFUNCTION(BlueprintCallable)
	void DebuggerHelp();
	
	// Sets default values for this actor's properties
	ADoor();

	// Called every frame
	virtual void Tick(float DeltaTime) override;

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	EntranceDirectionClass Direction;

	UPROPERTY(BlueprintReadWrite)
	bool CharacterWithinRange = false;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	TArray<AActor*> NewRoomActors;

	UPROPERTY(BlueprintReadWrite)
	bool Locked = false;


	TArray<FRenderingInformation> Segments;
	
	bool EnableInteraction = false;
	bool LoadNewRoom = true;

	std::vector<ObstacleInformation> RoomCells;

	FCriticalSection LoadNewRoomSection;
	FCriticalSection EnableInteractionSection;
	FCriticalSection SegmentsSection;
};
