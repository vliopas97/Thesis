// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "ObstacleBase.generated.h"

//Base class for Obstacles that have a fixed position
//Set to spawn from the ground (like spikes)
//Not used
UCLASS()
class MYPROJECT_API AObstacleBase : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	AObstacleBase();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	UFUNCTION(BlueprintImplementableEvent, BlueprintCallable)
	void RisingAnimation();

	UFUNCTION(BlueprintImplementableEvent, BlueprintCallable)
	void FallingAnimation();

};
