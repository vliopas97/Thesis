// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/StaticMeshComponent.h"
#include "HumanBase.generated.h"

UCLASS()
class MYPROJECT_API AHumanBase : public AActor
{
	GENERATED_BODY()
	 
public:	
	// Sets default values for this actor's properties
	AHumanBase();

	UFUNCTION(BlueprintImplementableEvent, BlueprintCallable)
	void RenderEffects();

	UPROPERTY(VisibleAnywhere)
	USceneComponent* Root;

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite)
	FVector2D DesiredPosition;


protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	virtual void OnConstruction(const FTransform& Transform) override;

private:

	void Rotate(float DeltaTime);

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

protected:

	static const int WalkingSpeed = 140; /*Walking Speed in cm/s */

	float RotationSpeed = 15;

	FRotator DesiredRotation = FRotator::ZeroRotator;
};
