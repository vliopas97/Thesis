include-debug - header files for debugging version of library
include-release = header files for release release version of library

!debug prints visual queues for the global map - release does not
!paste from {include-[debug|release]} to {include} folder to run debug or release version

lib/dGlobalMap.lib = lib file for debug
lib/rGlobalMap.lib = lib file for release

src = contains all the source files for building the project through an IDE
!NOTE CUDA required