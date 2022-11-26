// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "BasicTypes.h"
#include "Engine/GameInstance.h"
#include "TextBlock.h"
#include "Door.h"
#include "ObstacleBase.h"
#include "HumanBase.h"
#include "HumanObstacleAI.h"
#include "LoadingWidgetTemplate.h"
#include "Kismet/GameplayStatics.h"
#include "Tickable.h"
#include <shared_mutex>
#include "MyGameInstanceCode.generated.h"

/**
 * 
 */
UCLASS()
class MYPROJECT_API UMyGameInstanceCode : public UGameInstance, public FTickableGameObject
{
	GENERATED_BODY()

public:

	UMyGameInstanceCode(const FObjectInitializer& ObjectInitializer);

	virtual void Init() override;

	//Takes incoming frames from ZED camera and updates Global Map. Runs on a background thread
	UFUNCTION(BlueprintCallable, meta = (DisplayName = "Global Mapping", Keywords = "Track Space and update Global Map"))
	void GlobalMapping();

	//For rendering the first room upon booting the app
	UFUNCTION(BlueprintCallable, meta = (DisplayName = "Render Room"))
	void RenderRoomOnLoad(TArray<FRenderingInformation> Segments);

	UFUNCTION(BlueprintCallable)
	void RenderRoom(ADoor* Door);

	//For rendering the first room upon booting the up
	UFUNCTION(BlueprintCallable, meta = (DisplayName = "Get Room Information on Load", Keywords = "Track Space for New Room"))
	void GetRoomInformationOnLoad(TArray<FRenderingInformation>& Segments);

	UFUNCTION(BlueprintCallable, meta = (DisplayName = "Get Room Information", Keywords = "Track Space for New Room"))
	void GetRoomInformation(ADoor* Door);

	UFUNCTION(BlueprintCallable)
	void ClearPreviousRoom(ADoor* Door);

	UFUNCTION(BlueprintCallable)
	void ResetObstacles(ADoor* Door);

	UFUNCTION(BlueprintCallable)
	void AttachDoor(ADoor* Door);

	UFUNCTION(BlueprintCallable)
	void DetachDoor(ADoor* Door);

	UFUNCTION(BlueprintPure)
	bool GetBootLock();

	UFUNCTION(BlueprintPure)
	FTransform GetInitialTransform();

	UFUNCTION(BlueprintCallable)
	void ToggleVisibility(ADoor* Door, bool SameSide);

private:

	void ObstacleDetection(sl::Pose& Pose);

	void HumanDetection(std::vector<std::pair<int, sl::float2>>& result);

public:

	UPROPERTY(BlueprintReadWrite)
	TArray<AActor*> Actors;

	UPROPERTY(BlueprintReadWrite)
	TArray<FRenderingInformation> Segments;

	UPROPERTY(EditDefaultsOnly)
	TSubclassOf<AActor> WallClass;

	UPROPERTY(EditDefaultsOnly)
	TSubclassOf<ADoor> DoorClass;

	UPROPERTY(EditDefaultsOnly)
	TSubclassOf<AActor> FloorClass;

	UPROPERTY(EditDefaultsOnly)
	TSubclassOf<AActor> CeilingClass;

	//Enables the Object Detection part of the code that runs in GlobalMapping method
	UPROPERTY(EditDefaultsOnly)
	bool ObjectDetection = false;

	UPROPERTY(EditDefaultsOnly, BlueprintReadOnly)
	TSubclassOf<AObstacleBase> ObstacleClass;

	UPROPERTY(EditDefaultsOnly, BlueprintReadOnly)
	TSubclassOf<ACharacter> HumanClass;

	UPROPERTY(EditDefaultsOnly, BlueprintReadOnly)
	float WallHeight = 300;

	UFUNCTION(BlueprintCallable)
	virtual void Tick(float DeltaTime) override;

private:

	virtual void BeginDestroy() override;

	virtual bool IsTickable() const override;

	virtual TStatId GetStatId() const override;

	void AnimateObstacles(std::vector<ObstacleInformation>& RoomCells);

	void AnimateHumans();

	FTransform GetPlayerTransform();

	bool EndGameTrigger();

	void ReleaseMutexes();

	static sl::ERROR_CODE isFloorValid(const sl::float4& planeEquation,const sl::float3& position);

public:
	//Widgets
	ULoadingWidgetTemplate* LoadingWidget;
	ULoadingWidgetTemplate* ErrorWidget;

	std::vector<ObstacleInformation> RoomCells;
	//Stores list of Human Obstacles detected on each incoming camera frame
	std::vector<HumanInformation> Humans;

private:

	FTransform InitTransform;

	TSubclassOf<UUserWidget> LoadingWidgetClass;
	TSubclassOf<UUserWidget> ErrorWidgetClass;

	TSubclassOf<AActor> TorchClass;

	//Mutexes
	FCriticalSection DynamicObstacleDetection;
	FCriticalSection HumanObstacleDetection;
	FCriticalSection EndPlayLock;
	FCriticalSection InitTransformCrit;
	std::shared_mutex ReleaseMemoryLock;

	TPromise<void> BootRenderer;
	bool BootRendererRetrieved = false;
	FCriticalSection BootRendererCrit;

	bool EndPlay = false;
	bool bIsCreateOnRunning = true;

	//List of doors whose Bounding Box is being overlapped by player at the same time
	std::vector<ADoor*> DoorsOverlapped;

	friend class AFirstPersonBase;
	
};
