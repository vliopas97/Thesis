// Fill out your copyright notice in the Description page of Project Settings.

#include "Statue.h"
#include "UObject/ConstructorHelpers.h"
#include "Kismet/KismetMathLibrary.h"
#include "Kismet/GameplayStatics.h"
#include "GameFramework/Character.h"


AStatue::AStatue()
{
	StaticMesh = CreateDefaultSubobject<UStaticMeshComponent>("Static Mesh");
	StaticMesh->AttachTo(Root);
}

void AStatue::BeginPlay()
{
	Super::BeginPlay();
}

void AStatue::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	//Floating Effect
	RunningTime += DeltaTime;
	FVector Location = StaticMesh->RelativeLocation;
	float DeltaHeight = (FMath::Sin(RunningTime + DeltaTime) - FMath::Sin(RunningTime));
	Location.Z += DeltaHeight * 35.f;
	StaticMesh->SetRelativeLocation(Location);

	//Movement and Rotating Effect
	FVector CurrentPosition = GetActorLocation();
	auto DesiredPositionCp = DesiredPosition;

	if (DesiredPositionCp != FVector2D(CurrentPosition.X, CurrentPosition.Y))
	{
		SetActorLocation(FVector(FMath::Vector2DInterpConstantTo(
			FVector2D(CurrentPosition.X, CurrentPosition.Y), DesiredPositionCp, DeltaTime, WalkingSpeed), CurrentPosition.Z));
		DesiredRotation = UKismetMathLibrary::FindLookAtRotation(CurrentPosition, FVector(DesiredPositionCp, CurrentPosition.Z));
		SetActorRotation(FMath::RInterpConstantTo(GetActorRotation(), DesiredRotation, DeltaTime, 0.1 * abs(DesiredRotation.Yaw)));
	}
	else
	{
		auto CharacterLocation = UGameplayStatics::GetPlayerCharacter(GetWorld(), 0)->GetActorLocation();
		DesiredRotation = UKismetMathLibrary::FindLookAtRotation(CurrentPosition, FVector(CharacterLocation.X, CharacterLocation.Y, CurrentPosition.Z));
		SetActorRotation(FMath::RInterpConstantTo(GetActorRotation(), DesiredRotation, DeltaTime, 0.25 * abs(DesiredRotation.Yaw)));
	}

}
