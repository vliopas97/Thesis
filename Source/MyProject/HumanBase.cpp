// Fill out your copyright notice in the Description page of Project Settings.

#include "HumanBase.h"
#include "UObject/ConstructorHelpers.h"
#include "Kismet/KismetMathLibrary.h"
#include "Kismet/GameplayStatics.h"
#include "GameFramework/Character.h"

// Sets default values
AHumanBase::AHumanBase()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	Root = CreateDefaultSubobject<USceneComponent>("RootScene");
	RootComponent = Root;

}

// Called when the game starts or when spawned
void AHumanBase::BeginPlay()
{
	Super::BeginPlay();
	DesiredPosition = FVector2D(GetActorLocation().X, GetActorLocation().Y);
}

void AHumanBase::OnConstruction(const FTransform& Transform)
{
	Super::OnConstruction(Transform);
	//DesiredPosition = GetActorLocation();

	auto CurrentPosition = GetActorLocation();
	if (IsValid(UGameplayStatics::GetPlayerCharacter(GetWorld(), 0)))
	{
		auto CharacterLocation = UGameplayStatics::GetPlayerCharacter(GetWorld(), 0)->GetActorLocation();
		SetActorRotation(UKismetMathLibrary::FindLookAtRotation(CurrentPosition, FVector(CharacterLocation.X, CharacterLocation.Y, CurrentPosition.Z)));
	}
}

void AHumanBase::Rotate(float DeltaTime)
{
	FRotator CurrentRotation = GetActorRotation();
	if (CurrentRotation != DesiredRotation)
	{
		SetActorRotation(FMath::RInterpConstantTo(CurrentRotation, DesiredRotation, DeltaTime, RotationSpeed));
	}
}

// Called every frame
void AHumanBase::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
}

