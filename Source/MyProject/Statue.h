// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "HumanBase.h"
#include "Statue.generated.h"

/**
 * Basic class for representing a human obstacle
 * Renders a floating statue with no animation
 * For debug purposes
 */
UCLASS()
class MYPROJECT_API AStatue : public AHumanBase
{
	GENERATED_BODY()

public:

	AStatue();

	UPROPERTY(VisibleAnywhere)
	UStaticMeshComponent* StaticMesh;

protected:
	//Called when the game starts or when spawned
	virtual void BeginPlay() override;
	
public:
	//Called every frame
	virtual void Tick(float DeltaTime) override;

private:
	float RunningTime;
};
