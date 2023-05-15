// Fill out your copyright notice in the Description page of Project Settings.

#include "HumanObstacleAI.h"

// Sets default values
AHumanObstacleAI::AHumanObstacleAI()
{
 	// Set this character to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void AHumanObstacleAI::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void AHumanObstacleAI::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	FVector2D DesiredPositionCp = GetDesiredPosition();

	MoveTo(DesiredPositionCp);

}

// Called to bind functionality to input
void AHumanObstacleAI::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);

}

void AHumanObstacleAI::SetDesiredPosition(const FVector2D& DesPos)
{
	FScopeLock Lock(&PositionLock);
	DesiredPosition = DesPos;
}

FVector2D AHumanObstacleAI::GetDesiredPosition()
{
	FScopeLock Lock(&PositionLock);
	return DesiredPosition;
}

