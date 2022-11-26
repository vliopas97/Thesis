// Fill out your copyright notice in the Description page of Project Settings.

#include "MyProject.h"
#include "Modules/ModuleManager.h"
#include "Core/Public/Windows/WindowsPlatformProcess.h"
#include "Misc/Paths.h"
#include "sl/Camera.hpp"

void* DLLHandle = nullptr;

void FMyProject::StartupModule()
{
	auto ZED_SDK_Path = FWindowsPlatformMisc::GetEnvironmentVariable(TEXT("ZED_SDK_ROOT_DIR"));
	ZED_SDK_Path.ReplaceInline(TEXT("\\"), TEXT("/"));
	FPlatformProcess::AddDllDirectory(*ZED_SDK_Path);
	FString AbsPath = FPaths::Combine(*ZED_SDK_Path, TEXT("bin/"), TEXT("sl_zed64.dll"));
	DLLHandle = FPlatformProcess::GetDllHandle(*AbsPath);
}

void FMyProject::ShutdownModule()
{
	FPlatformProcess::FreeDllHandle(DLLHandle);
}

bool FMyProject::IsGameModule() const
{
	return true;
}

IMPLEMENT_PRIMARY_GAME_MODULE(FMyProject, MyProject, "MyProject");