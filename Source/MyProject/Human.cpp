// Fill out your copyright notice in the Description page of Project Settings.

#include "Human.h"
#include "UObject/ConstructorHelpers.h"
#include "Kismet/KismetMathLibrary.h"
#include "Kismet/GameplayStatics.h"
#include "GameFramework/Character.h"
#include "Animation/AnimBlueprint.h"
#include "GenericPlatform/GenericPlatformMath.h"

AHuman::AHuman()
{
	SkeletalMesh = CreateDefaultSubobject<USkeletalMeshComponent>("Skeletal Mesh");
	SkeletalMesh->AttachTo(Root);
	SkeletalMesh->SetEnableBodyGravity(true, NAME_None);

	auto SkeletalMeshAsset = ConstructorHelpers::FObjectFinder<USkeletalMesh>(TEXT("/Game/Mannequin/Character/Mesh/SK_Mannequin"));
	if (SkeletalMeshAsset.Object != nullptr)
	{
		SkeletalMesh->SetSkeletalMesh(SkeletalMeshAsset.Object);
	}

	static ConstructorHelpers::FObjectFinder<UAnimBlueprint> MeshAnimAsset(TEXT("/Game/AdvancedLocomotionV3/Characters/Mannequin/Mannequin_IK_AnimBP"));
	SkeletalMesh->AnimBlueprintGeneratedClass = MeshAnimAsset.Object->GetAnimBlueprintGeneratedClass();

}

void AHuman::CalculateMovement(float DeltaTime)
{
	CurrentPosition = GetTransform();

	//Calculate Distance Moved
	DistanceMoved = UKismetMathLibrary::VSize(FVector(CurrentPosition.GetLocation().X, CurrentPosition.GetLocation().Y, 0) - FVector(PreviousPosition.GetLocation().X, PreviousPosition.GetLocation().Y, 0));

	//Calculate Distance Rotated
	DistanceRotated = FGenericPlatformMath::Abs(CurrentPosition.Rotator().Yaw - PreviousPosition.Rotator().Yaw);

	//Discard very subtle movements
	if (DistanceMoved > MovementThreshold || DistanceRotated > RotationThreshold) 
	{
		//Calculate Direction of Movement
		Direction = PreviousPosition.GetRelativeTransform(CurrentPosition).Rotator().Yaw;
		UE_LOG(LogTemp, Warning, TEXT("Dir: %f"), Direction);

		//Calculate Speed
		Speed = (DistanceMoved + DistanceRotated) / (200.f * DeltaTime);
	}
	else
	{
		Speed = 0.f;
		Direction = 0.f;
	}
	//Update Actor Position
	PreviousPosition = CurrentPosition;

}

void AHuman::BeginPlay()
{
	Super::BeginPlay();
}

void AHuman::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	//Movement and Rotating Effect
	FVector StartingPosition = GetActorLocation();
	CurrentPosition = GetTransform();
	auto DesiredPositionCp = DesiredPosition;

	auto SetRotation = [this, StartingPosition, DeltaTime](const FVector& Location)
	{
		DesiredRotation = UKismetMathLibrary::FindLookAtRotation(StartingPosition, FVector(Location.X, Location.Y, StartingPosition.Z));
		auto T1 = FTransform();
		auto T2 = FTransform();
		T1.SetRotation(GetActorQuat());
		T2.SetRotation(FQuat(DesiredRotation));
		auto Difference = T2.GetRelativeTransform(T1).Rotator().Yaw;
		if (FGenericPlatformMath::Abs(Difference) > 20)
		{
			SetActorRotation(FMath::RInterpConstantTo(GetActorRotation(), DesiredRotation, DeltaTime, 60));
		}
		else
		{
			SetActorRotation(DesiredRotation);
		}
	};

	if (DesiredPositionCp != FVector2D(StartingPosition.X, StartingPosition.Y))
	{
		SetActorLocation(FVector(FMath::Vector2DInterpConstantTo(
			FVector2D(StartingPosition.X, StartingPosition.Y), DesiredPositionCp, DeltaTime, WalkingSpeed), StartingPosition.Z));
		//DesiredRotation = UKismetMathLibrary::FindLookAtRotation(StartingPosition, FVector(DesiredPositionCp, StartingPosition.Z));
		//SetActorRotation(FMath::RInterpConstantTo(GetActorRotation(), DesiredRotation, DeltaTime, 0.1 * abs(DesiredRotation.Yaw)));
		////SetActorRotation(DesiredRotation);
		SetRotation(FVector(DesiredPositionCp, StartingPosition.Z));
	}
	else
	{
		auto CharacterLocation = UGameplayStatics::GetPlayerPawn(GetWorld(), 0)->GetActorLocation();
		//DesiredRotation = UKismetMathLibrary::FindLookAtRotation(StartingPosition, FVector(CharacterLocation.X, CharacterLocation.Y, StartingPosition.Z));
		//SetActorRotation(FMath::RInterpConstantTo(GetActorRotation(), DesiredRotation, DeltaTime, 0.1 * abs(DesiredRotation.Yaw)));
		//SetActorRotation(DesiredRotation);
		SetRotation(FVector(CharacterLocation.X, CharacterLocation.Y, StartingPosition.Z));
	}

	CalculateMovement(DeltaTime);

	//CalculateMovement(DeltaTime);
}
