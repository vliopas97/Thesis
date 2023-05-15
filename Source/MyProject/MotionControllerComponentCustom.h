// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "MotionControllerComponent.h"
#include "MotionControllerComponentCustom.generated.h"

/**
 * 
 */
UCLASS()
class MYPROJECT_API UMotionControllerComponentCustom : public UMotionControllerComponent
{
	GENERATED_BODY()

	virtual void TickComponent(float DeltaTime, enum ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;
	
private:
	/** View extension object that can persist on the render thread without the motion controller component */
	class FViewExtension : public FSceneViewExtensionBase
	{
	public:
		FViewExtension(const FAutoRegister& AutoRegister, UMotionControllerComponentCustom* InMotionControllerComponent);
		virtual ~FViewExtension() {}

		/** ISceneViewExtension interface */
		virtual void SetupViewFamily(FSceneViewFamily& InViewFamily) override {}
		virtual void SetupView(FSceneViewFamily& InViewFamily, FSceneView& InView) override {}
		virtual void BeginRenderViewFamily(FSceneViewFamily& InViewFamily) override;
		virtual void PreRenderView_RenderThread(FRHICommandListImmediate& RHICmdList, FSceneView& InView) override {}
		virtual void PreRenderViewFamily_RenderThread(FRHICommandListImmediate& RHICmdList, FSceneViewFamily& InViewFamily) override;
		virtual void PostRenderViewFamily_RenderThread(FRHICommandListImmediate& RHICmdList, FSceneViewFamily& InViewFamily) override;
		virtual int32 GetPriority() const override { return -10; }
		virtual bool IsActiveThisFrame(class FViewport* InViewport) const;

	private:
		friend class UMotionControllerComponentCustom;

		/** Motion controller component associated with this view extension */
		UMotionControllerComponentCustom* MotionControllerComponent;
		FLateUpdateManager LateUpdate;
	};
	TSharedPtr< FViewExtension, ESPMode::ThreadSafe > ViewExtension;

	/** If true, the Position and Orientation args will contain the most recent controller state */
	bool PollControllerState(FVector& Position, FRotator& Orientation, float WorldToMetersScale);
};
