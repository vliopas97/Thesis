// Fill out your copyright notice in the Description page of Project Settings.

#include "VRPawn.h"
#include "UObject/ConstructorHelpers.h"
#include "Components/SceneComponent.h"
#include "Camera/CameraComponent.h"
#include "Components/SkeletalMeshComponent.h"
#include "Components/WidgetComponent.h"
#include "Components/StereoLayerComponent.h"
#include "HeadMountedDisplay/Public/MotionControllerComponent.h"
#include "MotionControllerComponentCustom.h"
#include "CameraComponentCustom.h"
#include "Animation/AnimBlueprint.h"
#include "HeadMountedDisplayFunctionLibrary.h"
#include "Async/Async.h"
#include "HeadMountedDisplayBase.h"
#include "Engine.h"
#include "OculusHMD/Public/OculusFunctionLibrary.h"
#include "OculusHMD/Private/OculusHMD.h"

class FOculusFunctionLibraryProxy : public UOculusFunctionLibrary {
	friend class AVRPawn;
};

// Sets default values
AVRPawn::AVRPawn()
{
 	// Set this pawn to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	DefaultSceneRoot = CreateDefaultSubobject<USceneComponent>("Default Scene Root");
	VRRoot = CreateDefaultSubobject<USceneComponent>("VR Root");
	VRRoot->AttachToComponent(DefaultSceneRoot, FAttachmentTransformRules::KeepRelativeTransform);

	Camera = CreateDefaultSubobject<UCameraComponentCustom>("Camera");
	Camera->bLockToHmd = true;

	Camera->AttachToComponent(VRRoot, FAttachmentTransformRules::KeepRelativeTransform);

	MotionController_L=CreateDefaultSubobject<UMotionControllerComponentCustom>("Motion Controller Left");
	MotionController_L->SetTrackingMotionSource("Left");
	MotionController_L->AttachToComponent(VRRoot, FAttachmentTransformRules::KeepRelativeTransform);
	
	MotionController_R=CreateDefaultSubobject<UMotionControllerComponentCustom>("Motion Controller Right");
	MotionController_R->SetTrackingMotionSource("Right");
	MotionController_R->AttachToComponent(VRRoot, FAttachmentTransformRules::KeepRelativeTransform);

	SkeletalMesh = CreateDefaultSubobject<USkeletalMeshComponent>("Skeletal Mesh");
	auto SkeletalMeshAsset = ConstructorHelpers::FObjectFinder<USkeletalMesh>(TEXT("/Game/Mannequin/Character/Mesh/SK_Mannequin"));
	if (SkeletalMeshAsset.Object != nullptr)
	{
		SkeletalMesh->SetSkeletalMesh(SkeletalMeshAsset.Object);
	}

	Widget = CreateDefaultSubobject<UWidgetComponent>("Widget");
	Widget->AttachToComponent(Camera, FAttachmentTransformRules::KeepRelativeTransform);
	Widget->SetRelativeLocationAndRotation(FVector(25, 0, 0), FRotator(0, 180, 0));
	Widget->SetRelativeScale3D(FVector(0.1, 0.05, 0.028125));
	Widget->SetDrawSize(FVector2D(2000, 2000));
	Widget->SetVisibility(false);
	
	StereoLayer = CreateDefaultSubobject<UStereoLayerComponent>("Stereo Layer");
	StereoLayer->AttachToComponent(Camera, FAttachmentTransformRules::KeepRelativeTransform);
	StereoLayer->SetRelativeLocation(FVector(25, 0, 0));
	StereoLayer->SetRelativeScale3D(FVector(0.1, 0.9, 0.50625));
	StereoLayer->bLiveTexture = true;
}

// Called when the game starts or when spawned
void AVRPawn::BeginPlay()
{
	Super::BeginPlay();
	
}
PRAGMA_DISABLE_OPTIMIZATION

void AVRPawn::GetTrackingInformation(FTransform& HMD, FTransform& LeftMotionController, FTransform& RightMotionController)
{
	FQuat DeviceRotation;
	FVector DevicePosition;
	static FTransform HMDInternal, LeftMotionControllerInternal, RightMotionControllerInternal;
	
	auto ControllerId = GetMotionControllersId();

	if (GEngine->XRSystem->GetCurrentPose(IXRTrackingSystem::HMDDeviceId, DeviceRotation, DevicePosition))
	{
		HMDInternal = FTransform(DeviceRotation, DevicePosition, FVector::OneVector);
	}

	if (GEngine->XRSystem->GetCurrentPose(ControllerId.Left, DeviceRotation, DevicePosition))
	{
		LeftMotionControllerInternal = FTransform(DeviceRotation, DevicePosition, FVector::OneVector);
	}

	if (GEngine->XRSystem->GetCurrentPose(ControllerId.Right, DeviceRotation, DevicePosition))
	{
		RightMotionControllerInternal = FTransform(DeviceRotation, DevicePosition, FVector::OneVector);
	}

	HMD = HMDInternal;
	LeftMotionController = LeftMotionControllerInternal;
	RightMotionController = RightMotionControllerInternal;
}

// Called every frame
void AVRPawn::Tick(float DeltaTime)
{
	//static auto InitLocation = SkeletalMesh->GetComponentLocation();
	Super::Tick(DeltaTime);

	if (DoOnce && IsValid(Widget->GetRenderTarget()))
	{
		StereoLayer->SetTexture(Widget->GetRenderTarget());
		DoOnce = false;
	}
	//static bool FirstTime = true;

	////Update head position from ZED
	//FScopeLock Lock(&TransformCrit);
	//auto Location = InternalTransform.GetLocation();
	//Location.Z = PlayerHeight;
	//Camera->SetWorldTransform(FTransform(InternalTransform.Rotator(), Location, FVector::OneVector));

	//Update controllers' world transforms here
	//SetControllersWorldTransform();

	//UpdateToWorldTransform();
}

// Called to bind functionality to input
void AVRPawn::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);

}

void AVRPawn::SetWidget(UUserWidget* WidgetInput)
{
	std::shared_lock<std::shared_mutex> Lock(ReleaseMemoryLock);
	
	AsyncTask(ENamedThreads::GameThread, [this, WidgetInput]()
		{
			if (IsValid(WidgetInput))
			{
				Widget->SetWidget(WidgetInput);
			}
		});
}

void AVRPawn::SetVisibility(bool bNewVisibility)
{
	std::shared_lock<std::shared_mutex> Lock(ReleaseMemoryLock);

	AsyncTask(ENamedThreads::GameThread, [this, bNewVisibility]()
		{
			Widget->SetVisibility(bNewVisibility);
			StereoLayer->SetVisibility(bNewVisibility);
		});
}

AVRPawn::MotionControllerId AVRPawn::GetMotionControllersId()
{
	int32 LeftHandID, RightHandID;
	FName DeviceName = GEngine->XRSystem->GetSystemName();
	
	//Oculus
	OculusHMD::FOculusHMD* HMD = FOculusFunctionLibraryProxy::GetOculusHMD();

	if (HMD)
	{
		LeftHandID  = HMD->GetDeviceId(EControllerHand::Left);
		RightHandID = HMD->GetDeviceId(EControllerHand::Right);
	}

	return MotionControllerId(LeftHandID, RightHandID);
}

void AVRPawn::SetControllersWorldTransform()
{
	//Transforms in Sensor Coordinate system
	FTransform HMD, LeftMotionController, RightMotionController;
	GetTrackingInformation(HMD, LeftMotionController, RightMotionController);

	//Update Motion Controllers Position
	//Controllers Tracked Position -> Sensor Space -> World Space
	//Order of calculations: (Origin->MotionController) = (Camera->MotionController) * (Origin->Camera)
	MotionController_L->SetWorldTransform(LeftMotionController.GetRelativeTransform(HMD) * Camera->K2_GetComponentToWorld());
	MotionController_R->SetWorldTransform(RightMotionController.GetRelativeTransform(HMD) * Camera->K2_GetComponentToWorld());
}

void AVRPawn::UpdateToWorldTransform()
{
	FTransform HMD, LeftMotionController, RightMotionController;
	GetTrackingInformation(HMD, LeftMotionController, RightMotionController);

	//static FTransform Ref;
	//
	//if (SetUpTransforms) {
	//	Ref = HMD;
	//	SetUpTransforms = false;
	//	FScopeLock Lock(&TransformCrit);
	//	Camera->SetWorldTransform(FTransform());
	//}
	//else {
	//	auto Transform = HMD.GetRelativeTransform(Ref);
	//	Transform.SetLocation(FVector(Transform.GetLocation().X, Transform.GetLocation().Y, Transform.GetLocation().Z + PlayerHeight));
	//	FScopeLock Lock(&TransformCrit);
	//	Camera->SetWorldTransform(Transform);
	//}

	MotionController_L->SetWorldTransform(LeftMotionController.GetRelativeTransform(HMD) * Camera->K2_GetComponentToWorld());
	MotionController_R->SetWorldTransform(RightMotionController.GetRelativeTransform(HMD) * Camera->K2_GetComponentToWorld());

	//Camera->SetWorldTransform(HMD);
	//MotionController_L->SetWorldTransform(LeftMotionController);
	//MotionController_R->SetWorldTransform(RightMotionController);
}

void AVRPawn::BeginDestroy()
{
	Super::BeginDestroy();
	std::unique_lock<std::shared_mutex> Lock(ReleaseMemoryLock);//Hold until all background threads referencing "this" are done
}

