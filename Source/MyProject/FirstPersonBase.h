// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Pawn.h"
#include <shared_mutex>
#include "Camera/CameraComponent.h"
#include "sl/Camera.hpp"
#include "FirstPersonBase.generated.h"

class UCameraComponentCustom;
class UMyGameInstanceCode;

//Replaces VRPawn and gets its tracking information through the ZED camera's tracking system
//Not used on release- ZED's tracking unreliable
//For debug purposes only
UCLASS()
class MYPROJECT_API AFirstPersonBase : public APawn
{
	GENERATED_BODY()

public:
	// Sets default values for this pawn's properties
	AFirstPersonBase();

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	USceneComponent* DefaultSceneRoot;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	UCameraComponent* Camera;

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	// Called to bind functionality to input
	virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

	//Updates actors Position and Orientation based on ZED's tracking
	//Called from UMyGameInstanceCode::GlobalMapping()
	void TrackBodyIK(sl::Pose& zed_pose, UMyGameInstanceCode* GameInstance);

private:

	virtual void BeginDestroy() override;

public:

	FCriticalSection TransformCrit;

	FTransform InternalTransform;
	

private:

	bool DoOnce = true;

	std::shared_mutex ReleaseMemoryLock;
};
