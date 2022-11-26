// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Blueprint/UserWidget.h"
#include "LoadingWidgetTemplate.generated.h"

/**
 * 
 */
UCLASS()
class MYPROJECT_API ULoadingWidgetTemplate : public UUserWidget
{
	GENERATED_BODY()
	
public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Text")
	FText Message;

	//UFUNCTION(BlueprintCallable)
	//FText GetDisplayText() const;

	//UFUNCTION(BlueprintCallable)
	//void SetDisplayText(const FText& NewDisplayText);
	
};
