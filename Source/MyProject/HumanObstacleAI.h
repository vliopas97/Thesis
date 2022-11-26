// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "HumanObstacleAI.generated.h"

UCLASS()
class MYPROJECT_API AHumanObstacleAI : public ACharacter
{
	GENERATED_BODY()

public:
	// Sets default values for this character's properties
	AHumanObstacleAI();

	UFUNCTION(BlueprintImplementableEvent, BlueprintCallable)
	void RenderEffects();

	UFUNCTION(BlueprintImplementableEvent, BlueprintCallable)
	void MoveTo(const FVector2D& Destination);
protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	// Called to bind functionality to input
	virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

	void SetDesiredPosition(const FVector2D& DesPos);

	FVector2D GetDesiredPosition();

private:

	FCriticalSection PositionLock;

	FVector2D DesiredPosition;

};
