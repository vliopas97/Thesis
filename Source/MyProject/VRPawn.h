// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Pawn.h"
#include <shared_mutex>
#include "VRPawn.generated.h"

class UCameraComponentCustom;
class UMotionControllerComponent;
class UMotionControllerComponentCustom;
class USceneComponent;
class USkeletalMeshComponent;
class UWidgetComponent;
class UStereoLayerComponent;

UCLASS()
class MYPROJECT_API AVRPawn : public APawn
{
	GENERATED_BODY()

	struct MotionControllerId {
		MotionControllerId(int32 Left = -1, int32 Right = -1) :
			Left(Left), Right(Right)
		{}

		int32 Left = -1;
		int32 Right = -1;
	};

public:
	// Sets default values for this pawn's properties
	AVRPawn();

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	USceneComponent* DefaultSceneRoot;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	USceneComponent* VRRoot;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	UCameraComponentCustom* Camera;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	UMotionControllerComponentCustom* MotionController_L;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	UMotionControllerComponentCustom* MotionController_R;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	USkeletalMeshComponent* SkeletalMesh;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	UWidgetComponent* Widget;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	UStereoLayerComponent* StereoLayer;

	UPROPERTY(EditDefaultsOnly, BlueprintReadOnly)
	float PlayerHeight = 180.f;

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override; 

private:
	//Gets Positional and Rotational tracking info from the Headset's sensors
	static void GetTrackingInformation(FTransform& HMD, FTransform& LeftMotionController, FTransform& RightMotionController);
	
	// Returns the ID for the tracked Motion Controller in the form (Left Motion Controller ID, Right Motion Controller ID)
	static MotionControllerId GetMotionControllersId();

	//To be called from tick
	//WARNING: Call after updating Camera World Location
	virtual void SetControllersWorldTransform();

	//Updates the controller's World Transform
	virtual void UpdateToWorldTransform();

	virtual void BeginDestroy() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	// Called to bind functionality to input
	virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

	//Called to attach widget
	inline void SetWidget(class UUserWidget* WidgetInput);

	//Remove any existing widgets
	inline void SetVisibility(bool bNewVisibility);

public:

	FCriticalSection TransformCrit;

	FTransform InternalTransform; //For Communicating with background threads of execution
	 
	//int32 LeftHandID = -1;
	//int32 RightHandID = -1;

private:
	bool DoOnce = true;

	std::shared_mutex ReleaseMemoryLock;
};
