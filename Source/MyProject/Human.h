// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "HumanBase.h"
#include "Components/SkeletalMeshComponent.h"
#include "Human.generated.h"

/**
 * 
 */
UCLASS()
class MYPROJECT_API AHuman : public AHumanBase
{
	GENERATED_BODY()
	
public:

	AHuman();

	UPROPERTY(VisibleAnywhere)
	USkeletalMeshComponent* SkeletalMesh;

	UFUNCTION(BlueprintCallable)
	void CalculateMovement(float DeltaTime);

protected:
	//Called when the games starts or when spawned
	virtual void BeginPlay() override;
	
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, meta = (AllowPrivateAccess = "true"))
	FTransform PreviousPosition;
	
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, meta = (AllowPrivateAccess = "true"))
	FTransform CurrentPosition;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, meta = (AllowPrivateAccess = "true"))
	float DistanceMoved;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, meta = (AllowPrivateAccess = "true"))
	float DistanceRotated;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, meta = (AllowPrivateAccess = "true"))
	float Speed;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, meta = (AllowPrivateAccess = "true"))
	float Direction;

public:
	//Called every frame
	virtual void Tick(float DeltaTime) override;

private:

	float MovementThreshold = 5.f;

	float RotationThreshold = 5.f;
};
