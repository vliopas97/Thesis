// Fill out your copyright notice in the Description page of Project Settings.

using UnrealBuildTool;
using System.IO;
using System;
using Tools.DotNETCommon;

public class MyProject : ModuleRules
{
	private string ModulePath { get { return ModuleDirectory; } }
	private string ThirdPartyPath { get { return Path.GetFullPath( Path.Combine( ModulePath, "../../ThirdParty/" ) ); } }

	public MyProject(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] {"Core", "CoreUObject", "Engine", "InputCore", "Niagara", "HeadMountedDisplay", "OculusHMD", "OVRPlugin" });

		PrivateDependencyModuleNames.AddRange(new string[] {"HeadMountedDisplay", "OculusHMD"});

        string engine_path = Path.GetFullPath(Target.RelativeEnginePath);

		PrivateIncludePathModuleNames.AddRange(new string[] {"Niagara", "HeadMountedDisplay", "OculusHMD"});

		string CudaSDKPath = System.Environment.GetEnvironmentVariable("CUDA_PATH_V9_0", EnvironmentVariableTarget.Machine);
        if (!Directory.Exists(CudaSDKPath))
            CudaSDKPath = System.Environment.GetEnvironmentVariable("CUDA_PATH_V10_0", EnvironmentVariableTarget.Machine);
        if (!Directory.Exists(CudaSDKPath))
            CudaSDKPath = System.Environment.GetEnvironmentVariable("CUDA_PATH_V10_2", EnvironmentVariableTarget.Machine);

		LoadZEDSDK(Target, System.Environment.GetEnvironmentVariable("ZED_SDK_ROOT_DIR", EnvironmentVariableTarget.Machine));
		LoadCUDA(Target, CudaSDKPath);
		LoadGM(Target);
        LoadMiscLibs(Target);

        // PublicDefinitions.Add("_SOLUTIONDIR=R\"($(SolutionDir))\"");
        
		// Uncomment if you are using Slate UI
		// PrivateDependencyModuleNames.AddRange(new string[] { "Slate", "SlateCore" });
		
		// Uncomment if you are using online features
		// PrivateDependencyModuleNames.Add("OnlineSubsystem");

		// To include OnlineSubsystemSteam, add it to the plugins section in your uproject file with the Enabled attribute set to true
	}

	 public void LoadCUDA(ReadOnlyTargetRules Target, string DirPath)
    {
        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            if (!Directory.Exists(DirPath))
            {
                string Err = string.Format("CUDA SDK missing");
                System.Console.WriteLine(Err);
                throw new BuildException(Err);
            }

            string[] LibrariesName =  {
                                        "cuda",
                                        "cudart",
                                        "cudadevrt"
                                      };

            PublicIncludePaths.Add(Path.Combine(DirPath, "include"));
            PublicLibraryPaths.Add(Path.Combine(DirPath, "lib\\x64"));

            foreach (string Library in LibrariesName)
            {
                PublicAdditionalLibraries.Add(Library + ".lib");
            }
        }
        else if (Target.Platform == UnrealTargetPlatform.Win32)
        {
            string Err = string.Format("Attempt to build against CUDA on unsupported platform {0}", Target.Platform);
            System.Console.WriteLine(Err);
            throw new BuildException(Err);
        }
    }

	public void LoadZEDSDK(ReadOnlyTargetRules Target, string DirPath)
    {
        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            if(!Directory.Exists(DirPath))
            {
                string Err = string.Format("ZED SDK missing");
                System.Console.WriteLine(Err);
                throw new BuildException(Err);
            }

            // Check SDK version
            string DefinesHeaderFilePath = Path.Combine(DirPath, "include\\sl\\Camera.hpp");
            string Major = "3";
            string Minor = "0";

            // Find SDK major and minor version and compare
            foreach (var line in File.ReadLines(DefinesHeaderFilePath))
            {
                if (!string.IsNullOrEmpty(line))
                {
                    if(line.Contains("#define ZED_SDK_MAJOR_VERSION"))
                    {
                        string SDKMajor = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries)[2];
                        if(!SDKMajor.Equals(Major))
                        {
                            string Err = string.Format("ZED SDK Major Version mismatch : found {0} expected {1}", SDKMajor, Major);
                            System.Console.WriteLine(Err);
                            throw new BuildException(Err);
                        }
                    }
                    else if (line.Contains("#define ZED_SDK_MINOR_VERSION"))
                    {
                        string SDKMinor = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries)[2];
                        if (!SDKMinor.Equals(Minor))
                        {
                            string Err = string.Format("ZED SDK Minor Version mismatch : found {0} expected {1}", SDKMinor, Minor);
                            System.Console.WriteLine(Err);
                            throw new BuildException(Err);
                        }

                        break;
                    }
                }
            }

            // Set the paths to the SDK
            string[] LibrariesNames = Directory.GetFiles(Path.Combine(DirPath, "lib"));
            string[] DynamicLibrariesNames = Directory.GetFiles(Path.Combine(DirPath, "bin"));

            PublicIncludePaths.Add(Path.Combine(DirPath, "include"));
            PublicLibraryPaths.Add(Path.Combine(DirPath, "lib"));
            PublicDelayLoadDLLs.Add(Path.Combine(DirPath, "bin", "sl_zed64.dll"));

            foreach (string Library in DynamicLibrariesNames)
            {
                PublicDelayLoadDLLs.Add(Library);
                // Log.TraceError(Library);
            }

            foreach (string Library in LibrariesNames)
            {
                PublicAdditionalLibraries.Add(Library);
            }
        }
        else if (Target.Platform == UnrealTargetPlatform.Win32)
        {
            string Err = string.Format("Attempt to build against ZED SDK on unsupported platform {0}", Target.Platform);
            System.Console.WriteLine(Err);
            throw new BuildException(Err);
        }
    }

	public void LoadGM(ReadOnlyTargetRules Target)
	{

		if (Target.Platform == UnrealTargetPlatform.Win64)
		{

			// string PlatformString = (Target.Platform == UnrealTargetPlatform.Win64) ? "x64" : "x86";
			string LibrariesPath = Path.Combine(ThirdPartyPath, "GM", "lib");

			/*
			test your path with:
			using System; // Console.WriteLine("");
			Console.WriteLine("... LibrariesPath -> " + LibrariesPath);
			*/

			PublicAdditionalLibraries.Add(Path.Combine(LibrariesPath, "rGlobalMap.lib")); 
			PublicIncludePaths.Add( Path.Combine( ThirdPartyPath, "GM", "include" ) );
			
			PublicDefinitions.Add(string.Format("WITH_GM_BINDING={0}", 1));
		}
		else
		{
	        string Err = string.Format("Attempt to build against CUDA on unsupported platform {0}", Target.Platform);
            System.Console.WriteLine(Err);
            throw new BuildException(Err);
		}

	}

    public void LoadMiscLibs(ReadOnlyTargetRules Target)
    {
        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            string BoostInclude = Path.Combine(ThirdPartyPath, "boost", "include");
            string OpenCVInclude = Path.Combine(ThirdPartyPath, "opencv", "include");
            string PythonInclude = Path.Combine(ThirdPartyPath, "python", "include");

            string BoostLib = Path.Combine(ThirdPartyPath, "boost", "lib");
            string EmbedderLib = Path.Combine(ThirdPartyPath, "embedder", "lib");
            string OpenCVLib = Path.Combine(ThirdPartyPath, "opencv", "lib");
            string PythonLib = Path.Combine(ThirdPartyPath, "python", "lib");

            PublicIncludePaths.Add(BoostInclude);
            PublicIncludePaths.Add(OpenCVInclude);
            PublicIncludePaths.Add(PythonInclude);

            PublicAdditionalLibraries.Add(Path.Combine(OpenCVLib, "opencv_core452.lib"));
            PublicAdditionalLibraries.Add(Path.Combine(OpenCVLib, "opencv_imgproc452.lib"));
            PublicAdditionalLibraries.Add(Path.Combine(OpenCVLib, "opencv_cudaimgproc452.lib"));
            PublicAdditionalLibraries.Add(Path.Combine(OpenCVLib, "opencv_highgui452.lib"));
            PublicAdditionalLibraries.Add(Path.Combine(OpenCVLib, "opencv_core452d.lib"));
            PublicAdditionalLibraries.Add(Path.Combine(OpenCVLib, "opencv_imgproc452d.lib"));
            PublicAdditionalLibraries.Add(Path.Combine(OpenCVLib, "opencv_cudaimgproc452d.lib"));
            PublicAdditionalLibraries.Add(Path.Combine(OpenCVLib, "opencv_highgui452d.lib"));
            PublicAdditionalLibraries.Add(Path.Combine(PythonLib, "python38.lib"));
            PublicLibraryPaths.Add(OpenCVLib);
            PublicDelayLoadDLLs.Add("opencv_core452.dll");
            PublicDelayLoadDLLs.Add("opencv_imgproc452.dll");
            PublicDelayLoadDLLs.Add("opencv_cudaimgproc452.dll");
            PublicDelayLoadDLLs.Add("opencv_highgui452.dll");
            PublicDelayLoadDLLs.Add("opencv_core452d.dll");
            PublicDelayLoadDLLs.Add("opencv_imgproc452d.dll");
            PublicDelayLoadDLLs.Add("opencv_cudaimgproc452d.dll");
            PublicDelayLoadDLLs.Add("opencv_highgui452d.dll");
            //PublicDelayLoadDLLs.Add("python38.dll");

            string[] LibrariesNames = Directory.GetFiles(BoostLib);
            foreach(string Library in LibrariesNames)
            {
                PublicAdditionalLibraries.Add(Library);
            }

            PublicDefinitions.Add("BOOST_DISABLE_ABI_HEADERS=1");
            PublicDefinitions.Add("WITH_OPENCV_BINDING=1");
            bUseRTTI = true;
            bEnableExceptions = true;
        }
        else
        {
            string Err = string.Format("Attempt to build on unsuported platform {}", Target.Platform);
            System.Console.WriteLine(Err);
            throw new BuildException(Err);
        }
    }
}
