// Fill out your copyright notice in the Description page of Project Settings.
#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB
#define HAVE_SNPRINTF
#include "MyGameInstanceCode.h"
#include "Blueprint/UserWidget.h"
#include "Kismet/GameplayStatics.h"
#include "UObject/ConstructorHelpers.h"
#include "Camera/CameraComponent.h"
#include "CameraComponentCustom.h"
#include "Components/WidgetComponent.h"
#include "Async/Async.h"
#include "TimerManager.h"
#include "VRPawn.h"
#include "FirstPersonBase.h"
#include <unordered_map>
#include <chrono>
#include <omp.h>

THIRD_PARTY_INCLUDES_START
#pragma push_macro("check")
#undef check
#include <boost/python.hpp>
#include <boost/filesystem.hpp>
#include <boost/python/numpy.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#pragma pop_macro("check")
THIRD_PARTY_INCLUDES_END

#include "globalMap.h"
#include "roomRenderer.h"
#include "utilities.h"

using namespace GM;
namespace py = boost::python;
namespace np = boost::python::numpy;


#define SCOPE_LOCK(LockName, CriticalSection)\
	{\
		FScopeLock LockName(&CriticalSection);

#define  SCOPE_UNLOCK\
	}\

#define EXCEPTIONCHECK(x)\
try{\
	x;\
}\
catch (boost::python::error_already_set&)\
{\
	PyObject* ptype, *pvalue, *ptraceback;\
	PyErr_Fetch(&ptype, &pvalue, &ptraceback);\
	if(pvalue)\
	{\
		PyObject* pstr = PyObject_Str(pvalue);\
		if(pstr)\
		{\
			const char* err_msg = PyUnicode_AsUTF8(pstr);\
			UE_LOG(LogTemp, Error, TEXT("EXCEPTION: %s"), *FString(err_msg));\
		}\
		PyErr_Restore(ptype, pvalue, ptraceback);\
	}\
}

namespace utils
{
	np::ndarray matToNDArray(const cv::Mat& image)
	{
		auto dtype = np::dtype::get_builtin<uchar>();
		py::tuple shape = py::make_tuple(image.rows, image.cols, image.channels());
		py::tuple stride = py::make_tuple(image.cols * image.channels() * sizeof(uchar), image.channels() * sizeof(uchar), sizeof(uchar));
		np::ndarray image_new = np::from_data(image.data, dtype, shape, stride, py::object());
		return image_new;
	}

	cv::Mat ndarrayToMat(const np::ndarray& nd)
	{
		const Py_intptr_t* shape = nd.get_shape();
		char* dtype_str = py::extract<char*>(py::str(nd.get_dtype()));

		unsigned int rows = shape[0];
		unsigned int cols = shape[1];
		unsigned int channels = shape[2];
		int depth;
		// find proper C++ type for the ndarray
		// in this case we use 'CV_8UC3'
		if (!strcmp(dtype_str, "uint8"))
		{
			depth = CV_8U;
		}
		else
		{
			throw "wrong dtype error";
		}

		int type = CV_MAKETYPE(depth, channels); // CV_8UC3

		auto size = cv::Size(cols, rows);
		cv::Mat mat = cv::Mat(size, type, reinterpret_cast<void*>(nd.get_data()));
		return mat;
	}

	int getOCVtype(sl::MAT_TYPE type)
	{
		int cv_type = -1;
		switch (type)
		{
		case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
		case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
		case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
		case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
		case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
		case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
		case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
		case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
		default: break;
		}
		return cv_type;
	}

	cv::Mat slMat2cvMat(sl::Mat& input)
	{
		cv::Mat image(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(sl::MEM::CPU), input.getStepBytes(sl::MEM::CPU));
		cv::cvtColor(image, image, cv::COLOR_BGRA2RGB);
		return image;
	}

	cv::cuda::GpuMat slMat2cvMatGPU(sl::Mat& input)
	{
		cv::cuda::GpuMat image(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(sl::MEM::GPU), input.getStepBytes(sl::MEM::GPU));
		cv::cuda::cvtColor(image, image, cv::COLOR_BGRA2RGB);
		return image;
	}

	std::vector<std::pair<int, sl::float2>> depthfinder(sl::Mat& depth, const std::vector<std::vector<float>>& result, float depthToImageRatio, sl::MEM memory_type = sl::MEM::CPU)
	{
		size_t predictions = result.size();

		std::vector<std::pair<int, sl::float2>> tracks;

		omp_lock_t lockguard;
		omp_init_lock(&lockguard);

#pragma omp parallel for if(predictions > 3)
		for (int i = 0; i < predictions; i++)
		{
			int trackID = result[i][0];
			int xmin = result[i][1] * depthToImageRatio;
			int ymin = result[i][2] * depthToImageRatio;
			int xmax = result[i][3] * depthToImageRatio;
			int ymax = result[i][4] * depthToImageRatio;

			std::unordered_map<float, float> xVals;
			std::unordered_map<float, float> yVals;

			for (int col = xmin; col < xmax; col++)
			{
				for (int row = ymin; row < ymax; row++)
				{
					sl::float4 point3d;
					depth.getValue(col, row, &point3d, memory_type);
					if (isnan(point3d.x) || isnan(point3d.y))
						continue;
					float x = round(point3d.x / 10.f);
					float y = round(point3d.y / 10.f);
					xVals[static_cast<int>(10 * x)]++;
					yVals[static_cast<int>(10 * y)]++;
				}
			}
			int max_countx = 0;
			int max_county = 0;
			int resx = -1;
			int resy = -1;
			auto xIter = xVals.begin();
			auto yIter = yVals.begin();

			for (size_t i = 0; i < xVals.size(); i++)
			{
				if (max_countx < xIter->second)
				{
					resx = xIter->first;
					max_countx = xIter->second;
				}
				xIter++;
			}
			for (size_t i = 0; i < yVals.size(); i++)
			{
				if (max_county < yIter->second)
				{
					resy = yIter->first;
					max_county = yIter->second;
				}
				yIter++;
			}
			omp_set_lock(&lockguard);
			//auto Pos = sl::float4(resx, resy, 0, 0) * PlayerTransform;
			tracks.emplace_back(std::make_pair(trackID, sl::float2(resx, resy)));//COORDS
			omp_unset_lock(&lockguard);
		}
		return tracks;
	}

}

class SSDNetwork 
{
public:
	SSDNetwork()
	{
		EXCEPTIONCHECK(module = py::import("ssd_final"));
		EXCEPTIONCHECK(py::exec("ssd=SSD()", module.attr("__dict__")));
		EXCEPTIONCHECK(ssd = module.attr("ssd"))
	}

	std::vector<std::vector<float>> predict(const cv::Mat& image)
	{
		CUcontext currentCudaContext;
		cuCtxPopCurrent(&currentCudaContext);
		auto input = utils::matToNDArray(image);
		auto result_ = ssd.attr("predict")(input);
		auto result = py::extract<np::ndarray>(result_).operator boost::python::numpy::ndarray();
		cuCtxPushCurrent(currentCudaContext);
		
		float* y_pred = reinterpret_cast<float*>(result.get_data());
		int rows = result.get_shape()[1];
		int cols = result.get_shape()[2];
		std::vector<std::vector<float>> output(rows, std::vector<float>(cols));
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				output[i][j] = (float)y_pred[i * cols + j];
			}
		}

		return output;
	}

	std::vector<std::vector<float>> predict(const cv::cuda::GpuMat& image)
	{
		auto ptr = (intptr_t)image.data;
		py::tuple shape = py::make_tuple(image.rows, image.cols, image.channels());
		py::tuple strides = py::make_tuple(image.step, image.channels() * sizeof(uchar), sizeof(uchar));
		py::object size(image.rows * image.cols * image.elemSize());

		CUcontext currentCudaContext;
		cuCtxPopCurrent(&currentCudaContext);
		np::ndarray result = py::extract<np::ndarray>(ssd.attr("predict_gpu")(ptr, shape, size, strides));
		cuCtxPushCurrent(currentCudaContext);
		float* y_pred = reinterpret_cast<float*>(result.get_data());

		int rows = result.get_shape()[1];
		int cols = result.get_shape()[2];
		std::vector<std::vector<float>> output(rows, std::vector<float>(cols));
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				output[i][j] = (float)y_pred[i * cols + j];
			}
		}

		return output;
	}

private:
	boost::python::api::object module;
	boost::python::api::object ssd;
};

namespace SSD
{
	void Initialize()
	{
		Py_Initialize();
		np::initialize();
		auto path = boost::filesystem::path("E:\\MyProject\\ThirdParty\\scripts\\");
		//path /= "ThirdParty\\scripts\\";
		FString s(path.generic_string().c_str());
		UE_LOG(LogTemp, Warning, TEXT("Path is: %s"), *s);
		PyObject* sys_path = PySys_GetObject("path");
		PyObject* name = PyUnicode_FromString(path.string().c_str());
		PyList_Append(sys_path, name);
		Py_DECREF(name);
	}

	std::vector<std::pair<int, sl::float2>> Predict(sl::Mat& Image, sl::Mat& DepthMap, sl::Transform World = sl::Transform())
	{
		static SSDNetwork ssd = SSDNetwork();
		assert(Image.getHeight() == Image.getWidth());
		assert(DepthMap.getHeight() == DepthMap.getWidth());
		assert(Image.getMemoryType() == sl::MEM::CPU);

		float depthToImageRatio = (float)DepthMap.getHeight() / (float)Image.getHeight();
		auto result = ssd.predict(utils::slMat2cvMat(Image));
		
		auto output = utils::depthfinder(DepthMap, result, depthToImageRatio);
		for (auto& out : output) {
			sl::float4 temp = sl::float4(out.second.x, out.second.y, 0, 1);
			temp = temp * World;
			out.second.x = temp.x;
			out.second.y = temp.y;
		}
		return output;
	}

	std::vector<std::pair<int, sl::float2>> Predict_GPU(sl::Mat& Image, sl::Mat& DepthMap)
	{
		static SSDNetwork ssd = SSDNetwork();
		assert(Image.getHeight() == Image.getWidth());
		assert(DepthMap.getHeight() == DepthMap.getWidth());
		assert(Image.getMemoryType() == sl::MEM::GPU);

		float depthToImageRatio = (float)DepthMap.getHeight() / (float)Image.getHeight();
		auto result = ssd.predict(utils::slMat2cvMatGPU(Image));
		return utils::depthfinder(DepthMap, result, depthToImageRatio, sl::MEM::GPU);
	}
}



UMyGameInstanceCode::UMyGameInstanceCode(const FObjectInitializer& ObjectInitializer)
{
	bIsCreateOnRunning = GIsRunning;

	static ConstructorHelpers::FClassFinder<UUserWidget> LoadingWidgetBlueprint(TEXT("/Game/Widgets/LoadingScreenWidget"));
	if (!ensure(LoadingWidgetBlueprint.Class != nullptr)) return;
	LoadingWidgetClass = LoadingWidgetBlueprint.Class;

	static ConstructorHelpers::FClassFinder<UUserWidget> ErrorWidgetBlueprint(TEXT("/Game/Widgets/ErrorMessageWidget"));
	if (!ensure(ErrorWidgetBlueprint.Class != nullptr)) return;
	ErrorWidgetClass = ErrorWidgetBlueprint.Class;

	static ConstructorHelpers::FClassFinder<AActor> TorchClassBlueprint(TEXT("/Game/Blueprints/Actors/Torch"));
	if (!ensure(TorchClassBlueprint.Class != nullptr)) return;
	TorchClass = TorchClassBlueprint.Class;

}

void UMyGameInstanceCode::Init()
{
	Super::Init();
	UE_LOG(LogTemp, Warning, TEXT("Found the Class named %s"), *LoadingWidgetClass->GetName());
	UE_LOG(LogTemp, Warning, TEXT("Found the Class named %s"), *ErrorWidgetClass->GetName());
	
	LoadingWidget = CreateWidget<ULoadingWidgetTemplate>(this, LoadingWidgetClass);
	ErrorWidget = CreateWidget<ULoadingWidgetTemplate>(this, ErrorWidgetClass);
}

PRAGMA_DISABLE_OPTIMIZATION
void UMyGameInstanceCode::GlobalMapping()
{
	auto Player = CastChecked<AVRPawn>(UGameplayStatics::GetPlayerPawn(GetWorld(), 0));
	Player->SetVisibility(true);
	
	//Add Viewport to Screen
	this->LoadingWidget->Message = FText::FromString("Looking for Camera Connection");
	//this->LoadingWidget->AddToViewport();
	Player->SetWidget(LoadingWidget);

	AsyncTask(ENamedThreads::AnyBackgroundHiPriTask, [this, Player]()
	{
		//FScopeLock DeleteLock(&this->Delete);
		std::shared_lock<std::shared_mutex> DeleteLock(this->ReleaseMemoryLock);
		bool ToBreak = EndGameTrigger();

		using namespace sl;
		sl::Camera zed;
		constexpr unsigned int imagedim = 300;
		constexpr unsigned int secondsToWait = 0;
		if(ObjectDetection)
			SSD::Initialize();

		sl::Mat pointCloud;
		sl::ERROR_CODE err;
		pointCloud.alloc(zed.getCameraInformation().camera_resolution, sl::MAT_TYPE::F32_C4, sl::MEM::GPU);
		sl::Mat ssdImage(imagedim, imagedim, sl::MAT_TYPE::U8_C4);
		sl::Mat ssdDepth(imagedim / 2, imagedim / 2, sl::MAT_TYPE::F32_C4);


		// Set configuration parameters for ZED
		sl::InitParameters init_params;
		init_params.coordinate_system = sl::COORDINATE_SYSTEM::LEFT_HANDED_Z_UP; // Use a right-handed Y-up coordinate system
		init_params.coordinate_units = sl::UNIT::CENTIMETER; // Set units in meters
		init_params.depth_mode = sl::DEPTH_MODE::ULTRA;
		init_params.camera_resolution = sl::RESOLUTION::HD720;
		init_params.camera_fps = 15;

		// Open the camera

		//if (this->ErrorWidget->IsInViewport())
		//{
		//	this->ErrorWidget->RemoveFromViewport();
		//}

		err = zed.open(init_params);
		//If Camera is not detected
		if (err != sl::ERROR_CODE::SUCCESS)
		{
			this->ErrorWidget->Message = FText::FromString("Camera not Detected. Please Restart...");
			//this->ErrorWidget->AddToViewport(1);
			//this->LoadingWidget->RemoveFromViewport();
			Player->SetWidget(ErrorWidget);
			FString code(errorCode2str(err).c_str());
			FString msg(sl::toVerbose(err).c_str());
			UE_LOG(LogTemp, Warning, TEXT("%s: %s"), *code, *msg);
			FPlatformProcess::SleepNoStats(3.f);
			ReleaseMutexes();
			return;
		}

		// Enable positional tracking with default parameters
		//this->LoadingWidget->Message = FText::FromString("Enabling Positional Tracking");
		sl::PositionalTrackingParameters tracking_parameters;
		tracking_parameters.enable_pose_smoothing = true;
		//tracking_parameters.initial_world_transform.setEulerAngles(sl::float3(0, 80, 0), false);

		//count = 0;
		//do
		//{
			err = zed.enablePositionalTracking(tracking_parameters);
		//	count++;
		//	FPlatformProcess::SleepNoStats(0.1);
		//} while (err != sl::ERROR_CODE::SUCCESS && count < 5);


		//If Camera's Positional Tracking cannot be enabled
		//if (err != sl::ERROR_CODE::SUCCESS)
		//{
		//	this->ErrorWidget->Message = FText::FromString("Positional Tracking Failed. Please Restart...");
		//	//this->ErrorWidget->AddToViewport(1);
		//	//this->LoadingWidget->RemoveFromViewport();
		//	Player->SetWidget(ErrorWidget);
		//	FPlatformProcess::SleepNoStats(3.f);
		//	ReleaseMutexes();
		//	return;
		//}
		this->LoadingWidget->Message = FText::FromString("Camera Connected. Please Wait.");

		//Setting up Camera's runtime parameters for image capturing
		sl::RuntimeParameters runtime_parameters;
		runtime_parameters.measure3D_reference_frame = REFERENCE_FRAME::CAMERA;
		runtime_parameters.sensing_mode = SENSING_MODE::FILL;

		//Capturing Camera's frames
		int i = 0;
		bool StartLevel = false;
		constexpr unsigned int ObjDetectionFrameFreq = 3;
		unsigned long ObjDetectionTrigger = ObjDetectionFrameFreq - 1;
		//sl::POSITIONAL_TRACKING_STATE tracking_state;

		ToBreak = EndGameTrigger();
		while (zed.getSVOPosition() <= zed.getSVONumberOfFrames() && !ToBreak)//comment out SVO condition on NDEBUG
		{
			auto start = std::chrono::high_resolution_clock::now();
			auto err = zed.grab(runtime_parameters);
			if (err == sl::ERROR_CODE::SUCCESS)
			{
				sl::Pose zed_pose;
				auto WorldTransform = GetPlayerTransform();//Taking Tracking Info from HMD

				//To comment out if tracking taken from the HMD
				//sl::POSITIONAL_TRACKING_STATE tracking_state;
				//tracking_state = zed.getPosition(zed_pose, REFERENCE_FRAME::WORLD);

				//if (tracking_state != sl::POSITIONAL_TRACKING_STATE::OK)
				//	continue;
				//auto WorldTransform = sl::unreal::ToUnrealType(sl::Transform(zed_pose.getOrientation(), zed_pose.getTranslation()));
				//float translation_left_to_center = zed.getCameraInformation().calibration_parameters.T.x * 0.5f;
				//GM::Utilities::transformPose(zed_pose.pose_data, translation_left_to_center);

				//ZED Tracking for Full Body Room Scale IK
				//CastChecked<AFirstPersonBase>(UGameplayStatics::GetPlayerPawn(GetWorld(), 0))->TrackBodyIK(zed_pose);
				//End of comment out if tracking taken from the HMD

				if (ObjectDetection && ++ObjDetectionTrigger == ObjDetectionFrameFreq)
				{
					zed.retrieveImage(ssdImage, sl::VIEW::LEFT, sl::MEM::CPU, sl::Resolution(ssdImage.getWidth(), ssdImage.getHeight()));
					zed.retrieveMeasure(ssdDepth, sl::MEASURE::XYZRGBA, sl::MEM::CPU, sl::Resolution(ssdDepth.getWidth(), ssdDepth.getHeight()));
					auto result = SSD::Predict(ssdImage, ssdDepth, sl::unreal::ToSlType(WorldTransform));
					if (result.size() > 0)
					{
						auto Pos = sl::unreal::ToUnrealType(result[0].second);
						UE_LOG(LogTemp, Error, TEXT("Frame No: %d prediction has track ID %d and position: %s"), i, result[0].first, *Pos.ToString());
					}
					else
					{
						UE_LOG(LogTemp, Error, TEXT("Frame No: %d"), i);
					}

					if (StartLevel)
					{
						this->HumanDetection(result);
					}
					ObjDetectionTrigger = 0;
				}

				//UE_LOG(LogTemp, Error, TEXT("Obj Detection Trigger No: %d"), ObjDetectionTrigger);
				if (i == secondsToWait * zed.getInitParameters().camera_fps)
				{
					this->BootRenderer.SetValue();
					StartLevel = true;
					//zed.enableRecording(RecordingParameters("E:/ZED/file.svo", SVO_COMPRESSION_MODE::H264));
				}
				
				//Update Height Map Procedure
				sl::Plane planeLocal;
				sl::Transform resetTrackingFloorFrame;
				//sl::ERROR_CODE ErrorCode = zed.findFloorPlane(planeLocal, resetTrackingFloorFrame/*floor prior = 180*/);
				
				//auto plane = GlobalMapDevice::transform(planeLocal.getPlaneEquation(), sl::unreal::ToSlType(WorldTransform));
				//
				//ErrorCode = isFloorValid(plane, sl::unreal::ToSlType(WorldTransform.GetLocation()));

				//if (ErrorCode == sl::ERROR_CODE::SUCCESS)
				//{
				//	zed.retrieveMeasure(pointCloud, MEASURE::XYZRGBA, MEM::GPU, sl::Resolution(1280, 720));
				//	GlobalMapDevice::transform(pointCloud, sl::unreal::ToSlType(WorldTransform));
				//	GlobalMapDevice::editGlobalMap(pointCloud, plane, sl::unreal::ToSlType(WorldTransform));
				//	
				//	//if(StartLevel)
				//		//this->ObstacleDetection(zed_pose);
				//}
				i++;
			}
			//else if (err == ERROR_CODE::END_OF_SVOFILE_REACHED)//to be removed/commented out on NDEBUG
			//{
			//	zed.setSVOPosition(0);
			//}
			else {
				FString code(errorCode2str(err).c_str());
				FString msg(sl::toVerbose(err).c_str());
				UE_LOG(LogTemp, Warning, TEXT("%s: %s"),*code, *msg);
			}
			ToBreak = EndGameTrigger();
		}

		//End of Process - Freeing up resources
		//zed.disableRecording();
		pointCloud.free(MEM::GPU);
		zed.disablePositionalTracking();
		zed.close();
		ReleaseMutexes();
	});

	AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask, [this]()
		{
			std::shared_lock<std::shared_mutex> DeleteLock(this->ReleaseMemoryLock);
			this->BootRenderer.GetFuture().Wait();
			FScopeLock Lock(&this->BootRendererCrit);
			BootRendererRetrieved = true;
		});
}

void UMyGameInstanceCode::RenderRoomOnLoad(TArray<FRenderingInformation> Segments)
{
	if (Segments.Num() == 0) return;

	float minX, minY, maxX, maxY;
	FActorSpawnParameters SpawnParameters;
	SpawnParameters.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
	
	minX = FGenericPlatformMath::Min(Segments[0].PointA.X, Segments[0].PointB.X);
	minY = FGenericPlatformMath::Min(Segments[0].PointA.Y, Segments[0].PointB.Y);
	maxX = FGenericPlatformMath::Max(Segments[0].PointA.X, Segments[0].PointB.X);;
	maxY = FGenericPlatformMath::Max(Segments[0].PointA.Y, Segments[0].PointB.Y);

	for (size_t i = 0; i < Segments.Num(); i++)
	{
		FVector2D Loc = (Segments[i].PointA + Segments[i].PointB) / 2;
		//FRotator Rotation(0, (Segments[i].PointA.X == Segments[i].PointB.X) ? 0 : 90, 0);
		FRotator Rotation;
		switch (Segments[i].DirectionInt)
		{
		case EntranceDirectionClass::MINUS_X:
			Rotation = FRotator::ZeroRotator;
			break;
		case EntranceDirectionClass::MINUS_Y:
			Rotation = FRotator(0, 90, 0);
			break;
		case EntranceDirectionClass::PLUS_X:
			Rotation = FRotator(0, 180, 0);
			break;
		case EntranceDirectionClass::PLUS_Y:
			Rotation = FRotator(0, 270, 0);
			break;
		}
		float length = ((Segments[i].PointA - Segments[i].PointB).Size()) / 100.f;
		FVector Scale = FVector(0.15, length, WallHeight / 100.0f);

		if (Segments[i].IsWall)
		{
			AActor* Object;
			Object = GetWorld()->SpawnActor<AActor>(WallClass, FVector(Loc, WallHeight/2.f), Rotation, SpawnParameters);
			Object->SetActorScale3D(Scale);
			Actors.Add(Object);

			AActor* Torch;
			Torch = GetWorld()->SpawnActor<AActor>(TorchClass, FVector(Loc, 0.8 * WallHeight / 2.f), Rotation, SpawnParameters);
			Actors.Add(Torch);
		}	
		else
		{	
			ADoor* Object;
			Object = Cast<ADoor>(UGameplayStatics::BeginDeferredActorSpawnFromClass(this, DoorClass, FTransform(Rotation, FVector(Loc, 0), FVector::OneVector)));
			if (Object)
			{
				Object->Direction = Segments[i].DirectionInt;
				Actors.Add(Object);
				UGameplayStatics::FinishSpawningActor(Object, FTransform(Rotation, FVector(Loc, 0), FVector::OneVector));
			}
		}
		
		if (Segments[i].PointA.X < minX)
		{
			minX = Segments[i].PointA.X;
		}
		else if (Segments[i].PointA.X > maxX)
		{
			maxX = Segments[i].PointA.X;
		}

		if (Segments[i].PointA.Y < minY)
		{
			minY = Segments[i].PointA.Y;
		}
		else if (Segments[i].PointA.Y > maxY)
		{
			maxY = Segments[i].PointA.Y;
		}


	}

	FVector4 v4(minX, minY, maxX, maxY);
	UE_LOG(LogTemp, Error, TEXT("minx, miny, maxx, maxy is: %s"), *v4.ToString());

	//for (float x = minX; x <= maxX; x += 64.f)
	//{
	//	for (float y = minY; y <= maxY; y += 64.f)
	//	{
	//		FTransform Transform(FRotator::ZeroRotator, FVector(x, y, 0), FVector::OneVector);

	//		auto TileObject = GetWorld()->SpawnActor<AActor>(TileFloorClass, FVector(x, y, 0), FRotator::ZeroRotator);

	//		Actors.Add(TileObject);
	//	}
	//}

	// Ceiling construction
	AActor* Ceiling;
	Ceiling = GetWorld()->SpawnActor<AActor>(CeilingClass, FVector((maxX + minX) / 2.f, (maxY + minY) / 2.f, WallHeight), FRotator::ZeroRotator, SpawnParameters);
	Ceiling->SetActorScale3D(FVector((maxX - minX) / 100.f, (maxY - minY) / 100.f, 1));
	Actors.Add(Ceiling);

	AActor* Floor;
	Floor = GetWorld()->SpawnActor<AActor>(FloorClass, FVector((maxX + minX) / 2.f, (maxY + minY) / 2.f, 0),
		FRotator::ZeroRotator, SpawnParameters);
	Floor->SetActorScale3D(FVector((maxX - minX) / 100.f, (maxY - minY) / 100.f, 1));
	Actors.Add(Floor);

	auto Player = CastChecked<AVRPawn>(UGameplayStatics::GetPlayerPawn(GetWorld(), 0));
	Player->SetVisibility(false);
}

void UMyGameInstanceCode::RenderRoom(ADoor * Door)
{
	TUniquePtr<FScopeLock> LockSegments = MakeUnique<FScopeLock>(&Door->SegmentsSection);
	//TArray<FRenderingInformation> Segments = Door->Segments;
	if (Door->Segments.Num() == 0)
		return;

	auto Pos = FVector2D(Door->GetActorLocation().X, Door->GetActorLocation().Y);
	auto Norm = FVector2D(Door->GetActorRightVector().X, Door->GetActorRightVector().Y);

	FActorSpawnParameters SpawnParameters;
	SpawnParameters.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;

	float minX = FGenericPlatformMath::Min(Door->Segments[0].PointA.X, Door->Segments[0].PointB.X);
	float minY = FGenericPlatformMath::Min(Door->Segments[0].PointA.Y, Door->Segments[0].PointB.Y);
	float maxX = FGenericPlatformMath::Max(Door->Segments[0].PointA.X, Door->Segments[0].PointB.X);;
	float maxY = FGenericPlatformMath::Max(Door->Segments[0].PointA.Y, Door->Segments[0].PointB.Y);

	for (size_t i = 0; i < Door->Segments.Num(); i++)
	{
		FVector2D Loc = (Door->Segments[i].PointA + Door->Segments[i].PointB) / 2;
		//FRotator Rotation(0, (Door->Segments[i].PointA.X == Door->Segments[i].PointB.X) ? 0 : 90, 0);
		FRotator Rotation;
		switch (Door->Segments[i].DirectionInt)
		{
		case EntranceDirectionClass::MINUS_X:
			Rotation = FRotator::ZeroRotator;
			break;
		case EntranceDirectionClass::MINUS_Y:
			Rotation = FRotator(0, 90, 0);
			break;
		case EntranceDirectionClass::PLUS_X:
			Rotation = FRotator(0, 180, 0);
			break;
		case EntranceDirectionClass::PLUS_Y:
			Rotation = FRotator(0, 270, 0);
			break;
		}
		float length = ((Door->Segments[i].PointA - Door->Segments[i].PointB).Size()) / 100.f;
		FVector Scale = FVector(0.15, length, WallHeight/100.0f);

		auto Tracer = [Loc, Pos, Norm](AActor* Object)
		{
			if (Norm.Y > Norm.X)
			{
				if (Loc.X == Pos.X)
				{
					Object->SetActorHiddenInGame(true);
				}
			}
			else
			{
				if (Loc.Y == Pos.Y)
				{
					Object->SetActorHiddenInGame(true);
				}
			}
		};

		if (Door->Segments[i].IsWall)
		{
			AActor* Object;
			Object = GetWorld()->SpawnActor<AActor>(WallClass, FVector(Loc, WallHeight/2.f), Rotation, SpawnParameters);
			Object->SetActorScale3D(Scale);
			Door->NewRoomActors.Add(Object);

			AActor* Torch;
			Torch = GetWorld()->SpawnActor<AActor>(TorchClass, FVector(Loc, 0.8 * WallHeight / 2.f), Rotation, SpawnParameters);
			Door->NewRoomActors.Add(Torch);
			//Tracer(Object);
		}
		else
		{
			ADoor* Object;
			Object = Cast<ADoor>(UGameplayStatics::BeginDeferredActorSpawnFromClass(this, DoorClass, FTransform(Rotation, FVector(Loc, 0), FVector::OneVector)));
			if (Object)
			{
				Object->Direction = Door->Segments[i].DirectionInt;
				Door->NewRoomActors.Add(Object);
				UGameplayStatics::FinishSpawningActor(Object, FTransform(Rotation, FVector(Loc, 0), FVector::OneVector));
			}
			//Tracer(Object);
		}

		if (Door->Segments[i].PointA.X < minX)
		{
			minX = Door->Segments[i].PointA.X;
		}
		else if (Door->Segments[i].PointA.X > maxX)
		{
			maxX = Door->Segments[i].PointA.X;
		}

		if (Door->Segments[i].PointA.Y < minY)
		{
			minY = Door->Segments[i].PointA.Y;
		}
		else if (Door->Segments[i].PointA.Y > maxY)
		{
			maxY = Door->Segments[i].PointA.Y;
		}

	}

	LockSegments.Reset();

	FVector4 v4(minX, minY, maxX, maxY);
	UE_LOG(LogTemp, Error, TEXT("minx, miny, maxx, maxy is: %s"), *v4.ToString());

	//for (float x = minX; x <= maxX; x += 64.f)
	//{
	//	for (float y = minY; y <= maxY; y += 64.f)
	//	{
	//		FTransform Transform(FRotator::ZeroRotator, FVector(x, y, 0), FVector::OneVector);

	//		auto TileObject = GetWorld()->SpawnActor<AActor>(FloorClass, FVector(x, y, 0), FRotator::ZeroRotator);

	//		Door->NewRoomActors.Add(TileObject);
	//	}
	//}

	// Ceiling construction
	AActor* Ceiling;
	Ceiling = GetWorld()->SpawnActor<AActor>(CeilingClass, FVector((maxX + minX) / 2.f, (maxY + minY) / 2.f, WallHeight), FRotator::ZeroRotator, SpawnParameters);
	Ceiling->SetActorScale3D(FVector((maxX - minX) / 100.f, (maxY - minY) / 100.f, 1));
	Door->NewRoomActors.Add(Ceiling);

	AActor* Floor;
	Floor = GetWorld()->SpawnActor<AActor>(FloorClass, FVector((maxX + minX) / 2.f, (maxY + minY) / 2.f, 0),
		FRotator::ZeroRotator, SpawnParameters);
	Floor->SetActorScale3D(FVector((maxX - minX) / 100.f, (maxY - minY) / 100.f, 1));
	Door->NewRoomActors.Add(Floor);
}

void UMyGameInstanceCode::GetRoomInformationOnLoad(TArray<FRenderingInformation>& Segments)
{
	auto InitTransform = GetInitialTransform();
	GM::roomRenderer Room(sl::unreal::ToSlType(FVector2D(InitTransform.GetLocation().X, InitTransform.GetLocation().Y)));
	Segments.Empty(Room.segments.size());
	auto p = Room.dInfo;

	auto toText = [](GM::FLOOR val) {
		if (val == GM::FLOOR::UNKNOWN) {
			return "U";
		}
		else if (val == GM::FLOOR::WALKABLE) {
			return "W";
		}
		else {
			return "O";
		}
	};

	for (int i = 0; i < p.size(); i++) {
		UE_LOG(LogTemp, Warning, TEXT("%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s"),
			toText(p[p.size() - i - 1][0].first),
			toText(p[p.size() - i - 1][1].first),
			toText(p[p.size() - i - 1][2].first),
			toText(p[p.size() - i - 1][3].first),
			toText(p[p.size() - i - 1][4].first),
			toText(p[p.size() - i - 1][5].first),
			toText(p[p.size() - i - 1][6].first),
			toText(p[p.size() - i - 1][7].first),
			toText(p[p.size() - i - 1][8].first),
			toText(p[p.size() - i - 1][9].first),
			toText(p[p.size() - i - 1][10].first),
			toText(p[p.size() - i - 1][11].first),
			toText(p[p.size() - i - 1][12].first),
			toText(p[p.size() - i - 1][13].first),
			toText(p[p.size() - i - 1][14].first),
			toText(p[p.size() - i - 1][15].first)
		);
	}

	for (auto& segment : Room.segments)
	{
		Segments.Emplace(segment);
	}

	FScopeLock ObstacleDetectionLock(&DynamicObstacleDetection);
	RoomCells.clear();
	for (auto& cell : Room.roomCells)
	{
		RoomCells.emplace_back(cell);
	}

	auto Player = CastChecked<AVRPawn>(UGameplayStatics::GetPlayerPawn(GetWorld(), 0));

	if (RoomCells.size() < 8)
	{
		ErrorWidget->Message = FText::FromString("Not enough space available for app to launch. Retrying...");

		//if (!ErrorWidget->IsInViewport())
		//	ErrorWidget->AddToViewport();
		Player->SetWidget(ErrorWidget);

	}
	else {
		Player->SetVisibility(false);
	}
	UE_LOG(LogTemp, Error, TEXT("SIZE: %d"), RoomCells.size());
	UE_LOG(LogTemp, Error, TEXT("SEGMENTS: %d"), Segments.Num());
	//LoadingWidget->RemoveFromViewport();
}

bool FindRoomCell(const sl::float2& slPosition, const GM::ENTRANCE_DIR& direction, const GM::GlobalMappingInformation& roomCell)
{
	switch (direction)
	{
	case GM::ENTRANCE_DIR::WALL:
		return false;
	case GM::ENTRANCE_DIR::X_DOWN:
		return (slPosition + sl::float2(-64, 64) == roomCell.position) ||
			(slPosition + sl::float2(-64, 0) == roomCell.position);
	case GM::ENTRANCE_DIR::X_UP:
		return (slPosition + sl::float2(64, -64) == roomCell.position) ||
			(slPosition + sl::float2(64, 0) == roomCell.position);
	case GM::ENTRANCE_DIR::Y_DOWN:
		return (slPosition + sl::float2(-64, -64) == roomCell.position) ||
			(slPosition + sl::float2(0, -64) == roomCell.position);
	case GM::ENTRANCE_DIR::Y_UP:
		return (slPosition + sl::float2(64, 64) == roomCell.position) ||
			(slPosition + sl::float2(0, 64) == roomCell.position);
	default:
		return false;
	}
}

void UMyGameInstanceCode::GetRoomInformation(ADoor * Door)
{
	AsyncTask(ENamedThreads::AnyBackgroundHiPriTask, [this, Door]()
		{
			FVector Position = Door->GetActorLocation();
			auto SlPosition = sl::unreal::ToSlType(FVector2D(Position.X, Position.Y));
			auto Direction = Utils::StripEntranceWrapper(Door->Direction);
			GM::roomRenderer room(SlPosition, Direction);
			auto p = room.dInfo;

			auto toText = [](GM::FLOOR val) {
				if (val == GM::FLOOR::UNKNOWN) {
					return "U";
				}
				else if (val == GM::FLOOR::WALKABLE) {
					return "W";
				}
				else {
					return "O";
				}
			};

			for (int i = 0; i < p.size(); i++) {
				UE_LOG(LogTemp, Warning, TEXT("%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s"),
					toText(p[p.size() - i - 1][0].first),
					toText(p[p.size() - i - 1][1].first),
					toText(p[p.size() - i - 1][2].first),
					toText(p[p.size() - i - 1][3].first),
					toText(p[p.size() - i - 1][4].first),
					toText(p[p.size() - i - 1][5].first),
					toText(p[p.size() - i - 1][6].first),
					toText(p[p.size() - i - 1][7].first),
					toText(p[p.size() - i - 1][8].first),
					toText(p[p.size() - i - 1][9].first),
					toText(p[p.size() - i - 1][10].first),
					toText(p[p.size() - i - 1][11].first),
					toText(p[p.size() - i - 1][12].first),
					toText(p[p.size() - i - 1][13].first),
					toText(p[p.size() - i - 1][14].first),
					toText(p[p.size() - i - 1][15].first)
				);
			}

			//512, 1472 slPosition
			TUniquePtr<FScopeLock> LockSegments = MakeUnique<FScopeLock>(&Door->SegmentsSection);

			Door->Segments.Empty(room.segments.size());
			for (auto& Element : room.segments)
			{
				Door->Segments.Emplace(Element);
			}
			//branch here
			if (Door->Segments.Num() < 2)
			{
				Door->Locked = true;
				UE_LOG(LogTemp, Error, TEXT("Locking due to no segments"));
			}
			UE_LOG(LogTemp, Error, TEXT("Segment size: %d"), Door->Segments.Num());
			LockSegments.Reset();

			TUniquePtr<FScopeLock> ObstacleDetectionLock = MakeUnique<FScopeLock>(&this->DynamicObstacleDetection);

			Door->RoomCells.reserve(room.roomCells.size());
			unsigned int counter = 0;
			for (auto& RoomCell : room.roomCells)
			{
				//counter = FindRoomCell(SlPosition, Direction, RoomCell) ? counter + 1 : counter;
				Door->RoomCells.emplace_back(RoomCell);
			}
			Door->Locked = Door->Locked || Door->RoomCells.size() < 4;
			if (Door->RoomCells.size() < 4) {
				UE_LOG(LogTemp, Error, TEXT("Locking due to no room"));
			}
			if (p[0][7].first != FLOOR::WALKABLE || p[0][8].first != FLOOR::WALKABLE || p[1][7].first != FLOOR::WALKABLE ||
				p[1][8].first != FLOOR::WALKABLE) 
			{
				Door->Locked = true;
			}

			ObstacleDetectionLock.Reset();

			FScopeLock LockLoadNewRoom(&Door->LoadNewRoomSection);
			FScopeLock LockEnableInteraction(&Door->EnableInteractionSection);
			Door->LoadNewRoom = true;
			Door->EnableInteraction = Door->CharacterWithinRange;
			UE_LOG(LogTemp, Warning, TEXT("Inside Critical Section (Background Thread"));
			UE_LOG(LogTemp, Error, TEXT("SIZE door: %d"), Door->RoomCells.size());
		});


}

void UMyGameInstanceCode::ClearPreviousRoom(ADoor* Door)
{
	auto& NewActors = Door->NewRoomActors;
	int32 Index = Actors.Find(Door);

	if (Index != INDEX_NONE)
	{
		NewActors.Add(Actors[Index]);
		Actors.RemoveAt(Index);
	}

	for (int i=0; i < Actors.Num(); i++)
	{
		if (IsValid(Actors[i]))
		{
			Actors[i]->Destroy();
		}
	}

	TUniquePtr<FScopeLock> ObstacleDetectionLock = MakeUnique<FScopeLock>(&DynamicObstacleDetection);//RoomCells is not Thread Safe - Mutex Lock
	for (auto& Room : RoomCells)
	{
		if (IsValid(Room.ActorPtr))
		{
			Room.ActorPtr->Destroy();
		}
	}

	RoomCells.clear();
	RoomCells = std::move(Door->RoomCells);
	ObstacleDetectionLock.Reset();

	Actors.Empty();

	Actors = MoveTemp(NewActors);

	DetachDoor(Door);
}

//void UMyGameInstanceCode::ResetObstacles()
//{
//	for (auto& Obstacle : RoomCellsTemp)
//	{
//		if (IsValid(Obstacle.ActorPtr))
//		{
//			Obstacle.ActorPtr->Destroy();
//		}
//	}
//	RoomCellsTemp.clear();
//}

void UMyGameInstanceCode::ResetObstacles(ADoor * Door)
{
	FScopeLock ObstacleDetectionLock(&this->DynamicObstacleDetection);
	for (auto& Obstacle : Door->RoomCells)
	{
		if (IsValid(Obstacle.ActorPtr))
		{
			Obstacle.ActorPtr->Destroy();
		}
	}
	Door->RoomCells.clear();
	DetachDoor(Door);
}

namespace
{
	template <typename Container, typename Functor>
	void for_each_indexed(Container& c, Functor f)
	{
		for (auto& e : c)
			f(e);
	}
}

void UMyGameInstanceCode::Tick(float DeltaTime)
{
	FActorSpawnParameters SpawnParameters;
	SpawnParameters.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::Undefined;

	TUniquePtr<FScopeLock> ObstacleDetectionLock = MakeUnique<FScopeLock>(&DynamicObstacleDetection);//RoomCells is not Thread Safe - Mutex Lock

	//AnimateObstacles(RoomCells);

	//for (auto& Door : DoorsOverlapped)
	//{
	//	AnimateObstacles(Door->RoomCells);
	//}

	ObstacleDetectionLock.Reset();

	AnimateHumans();
}

void UMyGameInstanceCode::AttachDoor(ADoor * Door)
{
	FScopeLock ObstacleDetectionLock(&this->DynamicObstacleDetection);
	DoorsOverlapped.emplace_back(Door);
}

void UMyGameInstanceCode::BeginDestroy()
{
	Super::BeginDestroy();
	TUniquePtr<FScopeLock> Lock = MakeUnique<FScopeLock>(&this->EndPlayLock);
	EndPlay = true;
	Lock.Reset();

	//FScopeLock DeleteLock(&Delete);
	std::unique_lock<std::shared_mutex> DeleteLock(ReleaseMemoryLock);
	
	FScopeLock Lock2(&BootRendererCrit);
	if (!BootRendererRetrieved)
		BootRenderer.SetValue();

	//if(IsValid(LoadingWidget)) LoadingWidget->RemoveFromParent();
	//if(IsValid(ErrorWidget)) ErrorWidget->RemoveFromParent();
}

bool UMyGameInstanceCode::IsTickable() const
{
	return bIsCreateOnRunning;
}

TStatId UMyGameInstanceCode::GetStatId() const
{
	return UObject::GetStatID();
}

void UMyGameInstanceCode::AnimateObstacles(std::vector<ObstacleInformation>& RoomCells)
{
	for (auto& RoomCell : RoomCells)
	{
		//if counter 0->positive value in this frame then spawn actor or perform rising animation
		if (RoomCell.GlobalMappingInformation.counter > 0 && RoomCell.CounterPrevFrame == 0)
		{
			if (!IsValid(RoomCell.ActorPtr))
			{
				RoomCell.ActorPtr = GetWorld()->SpawnActor<AObstacleBase>(ObstacleClass, FVector(RoomCell.Position, -100), FRotator(0, 0, 0));
			}
			RoomCell.ActorPtr->RisingAnimation();
		}
		else if (RoomCell.GlobalMappingInformation.counter == 0 && RoomCell.CounterPrevFrame > 0)
		{
			if (IsValid(RoomCell.ActorPtr))
			{
				RoomCell.ActorPtr->FallingAnimation();
			}
		}
	}
}

void UMyGameInstanceCode::AnimateHumans()
{
	FScopeLock Lock(&HumanObstacleDetection);
	for (int i = 0; i < Humans.size(); i++)
	{
		if (Humans[i].TrackState == HumanInformation::State::New)
		{
			Humans[i].ActorPtr = GetWorld()->SpawnActor<AHumanObstacleAI>(HumanClass, FVector(Humans[i].Position, 120), FRotator::ZeroRotator);
			Humans[i].ActorPtr->RenderEffects();
			Humans[i].TrackState = HumanInformation::State::Active;
			continue;
		}
		else if (Humans[i].TrackState == HumanInformation::State::PendingDestruction)
		{
			Humans[i].ActorPtr->Destroy();
			Humans.erase(Humans.begin() + i--);
			continue;
		}
		//Humans[i].ActorPtr->DesiredPosition = Humans[i].Position;
		Humans[i].ActorPtr->SetDesiredPosition(Humans[i].Position);
	}

}

FTransform UMyGameInstanceCode::GetPlayerTransform()
{
	//auto PlayerCharacter = CastChecked<AVRPawn>(UGameplayStatics::GetPlayerPawn(GetWorld(), 0));
	auto PlayerCharacter = Cast<AVRPawn>(UGameplayStatics::GetPlayerPawn(GetWorld(), 0));

	if (IsValid(PlayerCharacter)) {
		FScopeLock Lock(&PlayerCharacter->TransformCrit);
		return PlayerCharacter->Camera->GetComponentToWorld();
	}
	return FTransform();
}

bool UMyGameInstanceCode::EndGameTrigger()
{
	FScopeLock BreakLock(&this->EndPlayLock);
	return this->EndPlay;
}

void UMyGameInstanceCode::ReleaseMutexes()
{
	FScopeLock Lock(&BootRendererCrit);
	auto local = BootRendererRetrieved;
	if (!local)
	{
		BootRenderer.SetValue();
	}
}

sl::ERROR_CODE UMyGameInstanceCode::isFloorValid(const sl::float4& planeEquation,const sl::float3& position)
{
	float Dist = abs((planeEquation.x * position.x + planeEquation.y * position.y + planeEquation.z * position.z + planeEquation.w) /
		sqrt(planeEquation.x * planeEquation.x + planeEquation.y * planeEquation.y + planeEquation.z * planeEquation.z));
	auto Phi = atan(sqrt(planeEquation.x * planeEquation.x + planeEquation.y * planeEquation.y) / planeEquation.z);

	Phi = (180.f / 3.141559265) * Phi;

	if (Dist < 130.f || !std::isfinite(Dist) || abs(Phi) > 15.f)
		return sl::ERROR_CODE::PLANE_NOT_FOUND;
	return sl::ERROR_CODE::SUCCESS;
}

void UMyGameInstanceCode::DetachDoor(ADoor * Door)
{
	auto iterator = std::find(DoorsOverlapped.begin(), DoorsOverlapped.end(), Door);
	if (iterator != DoorsOverlapped.end())
	{
		FScopeLock ObstacleDetectionLock(&this->DynamicObstacleDetection);
		DoorsOverlapped.erase(iterator);
	}
}

bool UMyGameInstanceCode::GetBootLock()
{
	FScopeLock Lock(&BootRendererCrit);
	return BootRendererRetrieved;
}

FTransform UMyGameInstanceCode::GetInitialTransform()
{
	FScopeLock Lock(&InitTransformCrit);
	return InitTransform;
}

void UMyGameInstanceCode::ToggleVisibility(ADoor* Door, bool SameSide)
{
	auto Pos = FVector2D(Door->GetActorLocation().X, Door->GetActorLocation().Y);
	auto Norm = FVector2D(Door->GetActorRightVector().X, Door->GetActorRightVector().Y);
	auto& ToReveal = SameSide ? Actors : Door->NewRoomActors;
	auto& ToHide = SameSide ? Door->NewRoomActors : Actors;

	for (auto& Actor : ToReveal)
	{
		if (Actor->bHidden)
			Actor->SetActorHiddenInGame(false);
	}
	for (auto& Actor : ToHide)
	{
		if (Actor == Door)
			continue;

		auto Loc = FVector2D(Actor->GetActorLocation().X, Actor->GetActorLocation().Y);
		if (Norm.Y > Norm.X)
		{
			if (Loc.X == Pos.X)
			{
				Actor->SetActorHiddenInGame(true);
			}
		}
		else
		{
			if (Loc.Y == Pos.Y)
			{
				Actor->SetActorHiddenInGame(true);
			}
		}
	}
}

void UMyGameInstanceCode::ObstacleDetection(sl::Pose& Pose)
{
	//auto Scanner = [](std::vector<ObstacleInformation>& RoomCells)
	//{
	//	std::vector<GM::GlobalMappingInformation> Replica;
	//	Replica.reserve(RoomCells.size());
	//	for (auto& RoomCell : RoomCells)
	//	{
	//		Replica.emplace_back(RoomCell.GlobalMappingInformation);
	//		RoomCell.CounterPrevFrame = RoomCell.GlobalMappingInformation.counter;
	//	}
	//	
	//	GlobalMapDevice::scanForObstacles(Replica);
	//
	//	for (int i = 0; i < Replica.size(); i++)
	//	{
	//		RoomCells[0].GlobalMappingInformation = Replica[0];
	//	}
	//};
	//
	//FScopeLock GameInstanceLock(&DynamicObstacleDetection);//RoomCells are not Thread Safe - Mutex Lock
	//UpdateRoomCells(Pose);
	//
	//Scanner(RoomCells);
	//for (auto& Door : DoorsOverlapped)
	//{
	//	Scanner(Door->RoomCells);
	//}

	auto StripVector = [](std::vector<ObstacleInformation>& input)
	{
		std::vector<GM::GlobalMappingInformation> output;
		output.reserve(input.size());
		for (auto& in : input)
		{
			in.CounterPrevFrame = in.GlobalMappingInformation.counter;
			output.emplace_back(in.GlobalMappingInformation);
		}
		return output;
	};

	auto UpdateVector = [](const std::vector<GM::GlobalMappingInformation>& input, std::vector<ObstacleInformation>& output)
	{
		if (input.size() != output.size())
			throw "Obstacle Detection: Room Cells sizes mismatch";

		for (int i = 0; i < input.size(); i++)
		{
			output[i].GlobalMappingInformation = input[i];
		}
	};

	FScopeLock ObstacleDetectionLock(&this->DynamicObstacleDetection);

	std::vector<GM::GlobalMappingInformation> Replica;

	std::vector<int> indices;
	int size = 0;
	indices.emplace_back(size);
	size += RoomCells.size();
	for (auto& Door : DoorsOverlapped)
	{
		indices.emplace_back(Door->RoomCells.size());
		size += Door->RoomCells.size();
	}

	Replica.reserve(size);

	int counter = 0;
	auto vec = StripVector(RoomCells);
	Replica.insert(Replica.end(), vec.begin(), vec.end());
	for (auto& Door : DoorsOverlapped)
	{
		vec = StripVector(Door->RoomCells);
		Replica.insert(Replica.end(), vec.begin(), vec.end());
	}

	indices.emplace_back(Replica.size());

	GlobalMapDevice::scanForObstacles(Replica);

	int i = 0;
	UpdateVector(std::move(std::vector<GM::GlobalMappingInformation>(std::make_move_iterator(Replica.begin() + indices[i]),
		std::make_move_iterator(Replica.begin() + indices[++i]))), RoomCells);
	for (auto& Door : DoorsOverlapped)
	{
		UpdateVector(std::move(std::vector<GM::GlobalMappingInformation>(std::make_move_iterator(Replica.begin() + indices[i]), 
			std::make_move_iterator(Replica.begin() + indices[++i]))), Door->RoomCells);
	}

}

void UMyGameInstanceCode::HumanDetection(std::vector<std::pair<int, sl::float2>>& result)
{
	FScopeLock Lock(&HumanObstacleDetection);
	for (auto& Human : Humans)
	{
		const auto it = std::find_if(result.begin(), result.end(), [&Human](const std::pair<int, sl::float2>& Pair)
			{
				return Pair.first == Human.TrackID;
			});

		if (it == result.end())
		{
			Human.TrackState = HumanInformation::State::PendingDestruction;
			continue;
		}

		Human.Position = sl::unreal::ToUnrealType(result.at(it - result.begin()).second);
		result.erase(it);
	}

	for (auto& Element : result)
	{
		Humans.emplace_back();
		Humans.back().TrackID = Element.first;
		Humans.back().Position = sl::unreal::ToUnrealType(Element.second);
	}
}
