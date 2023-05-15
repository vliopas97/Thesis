// Fill out your copyright notice in the Description page of Project Settings.

#include "CameraComponentCustom.h"
#include "GameFramework/Controller.h"
#include "Engine/Engine.h"
#include "IHeadMountedDisplay.h"
#include "IXRTrackingSystem.h"
#include "IXRCamera.h"
#include "VRPawn.h"
#include "Components/WidgetComponent.h"

void UCameraComponentCustom::GetCameraView(float DeltaTime, FMinimalViewInfo& DesiredView)
{
	auto VRPawn = CastChecked<AVRPawn>(GetAttachmentRootActor());
	if (GEngine && GEngine->XRSystem.IsValid() && GetWorld() && GetWorld()->WorldType != EWorldType::Editor)
	{
		IXRTrackingSystem* XRSystem = GEngine->XRSystem.Get();
		auto XRCamera = XRSystem->GetXRCamera();

		if (XRCamera.IsValid())
		{
			if (XRSystem->IsHeadTrackingAllowed())
			{
				const FTransform ParentWorld = CalcNewComponentToWorld(FTransform());

				XRCamera->SetupLateUpdate(ParentWorld, this, bLockToHmd == 0);

				if (bLockToHmd)
				{
					FQuat Orientation;
					FVector Position;
					if (XRCamera->UpdatePlayerCamera(Orientation, Position))
					{
						if (SetUpTransforms) 
						{
							auto Rot = FRotator(Orientation);
							Rot.Pitch = 0.f;
							Rot.Roll = 0.f;
							Ref = FTransform(Rot, Position);
							SetUpTransforms = false;
							FScopeLock(&VRPawn->TransformCrit);
							SetRelativeTransform(FTransform());
							SetWorldTransform(FTransform());
						}
						else 
						{
							auto Transform = FTransform(Orientation, Position).GetRelativeTransform(Ref);
							Transform.SetLocation(FVector(Transform.GetLocation().X, Transform.GetLocation().Y, Transform.GetLocation().Z + VRPawn->PlayerHeight));
							FScopeLock(&VRPawn->TransformCrit);
							SetWorldTransform(Transform);
						}
						auto Player = CastChecked<AVRPawn>(GetOwner());
						Player->Widget->SetRelativeLocationAndRotation(FVector(20, 0, 0), FRotator(0, 180, 0));
					}
					else
					{
						ResetRelativeTransform();
					}
				}

				XRCamera->OverrideFOV(this->FieldOfView);
			}
		}
	}

	if (bUsePawnControlRotation)
	{
		const APawn* OwningPawn = Cast<APawn>(GetOwner());
		const AController* OwningController = OwningPawn ? OwningPawn->GetController() : nullptr;
		if (OwningController && OwningController->IsLocalPlayerController())
		{
			const FRotator PawnViewRotation = OwningPawn->GetViewRotation();
			if (!PawnViewRotation.Equals(GetComponentRotation()))
			{
				SetWorldRotation(PawnViewRotation);
			}
		}
	}

	if (bUseAdditiveOffset)
	{
		FTransform OffsetCamToBaseCam = AdditiveOffset;
		FTransform BaseCamToWorld = GetComponentToWorld();
		FTransform OffsetCamToWorld = OffsetCamToBaseCam * BaseCamToWorld;

		DesiredView.Location = OffsetCamToWorld.GetLocation();
		DesiredView.Rotation = OffsetCamToWorld.Rotator();
	}
	else
	{
		DesiredView.Location = GetComponentLocation();
		DesiredView.Rotation = GetComponentRotation();
	}

	DesiredView.FOV = bUseAdditiveOffset ? (FieldOfView + AdditiveFOVOffset) : FieldOfView;
	DesiredView.AspectRatio = AspectRatio;
	DesiredView.bConstrainAspectRatio = bConstrainAspectRatio;
	DesiredView.bUseFieldOfViewForLOD = bUseFieldOfViewForLOD;
	DesiredView.ProjectionMode = ProjectionMode;
	DesiredView.OrthoWidth = OrthoWidth;
	DesiredView.OrthoNearClipPlane = OrthoNearClipPlane;
	DesiredView.OrthoFarClipPlane = OrthoFarClipPlane;

	// See if the CameraActor wants to override the PostProcess settings used.
	DesiredView.PostProcessBlendWeight = PostProcessBlendWeight;
	if (PostProcessBlendWeight > 0.0f)
	{
		DesiredView.PostProcessSettings = PostProcessSettings;
	}

	DesiredView.CameraRenderingSettings = CameraRenderingSettings;
}
