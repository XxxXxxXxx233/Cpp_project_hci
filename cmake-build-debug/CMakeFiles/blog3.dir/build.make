# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = C:\Users\XX\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\201.7223.86\bin\cmake\win\bin\cmake.exe

# The command to remove a file.
RM = C:\Users\XX\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\201.7223.86\bin\cmake\win\bin\cmake.exe -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\XX\CLionProjects\hci_project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\XX\CLionProjects\hci_project\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/blog3.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/blog3.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/blog3.dir/flags.make

CMakeFiles/blog3.dir/blog3.cpp.obj: CMakeFiles/blog3.dir/flags.make
CMakeFiles/blog3.dir/blog3.cpp.obj: CMakeFiles/blog3.dir/includes_CXX.rsp
CMakeFiles/blog3.dir/blog3.cpp.obj: ../blog3.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\XX\CLionProjects\hci_project\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/blog3.dir/blog3.cpp.obj"
	C:\MinGW_W64\mingw64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\blog3.dir\blog3.cpp.obj -c C:\Users\XX\CLionProjects\hci_project\blog3.cpp

CMakeFiles/blog3.dir/blog3.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/blog3.dir/blog3.cpp.i"
	C:\MinGW_W64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\XX\CLionProjects\hci_project\blog3.cpp > CMakeFiles\blog3.dir\blog3.cpp.i

CMakeFiles/blog3.dir/blog3.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/blog3.dir/blog3.cpp.s"
	C:\MinGW_W64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\XX\CLionProjects\hci_project\blog3.cpp -o CMakeFiles\blog3.dir\blog3.cpp.s

# Object files for target blog3
blog3_OBJECTS = \
"CMakeFiles/blog3.dir/blog3.cpp.obj"

# External object files for target blog3
blog3_EXTERNAL_OBJECTS =

blog3.exe: CMakeFiles/blog3.dir/blog3.cpp.obj
blog3.exe: CMakeFiles/blog3.dir/build.make
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_gapi420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_stitching420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_aruco420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_bgsegm420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_bioinspired420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_ccalib420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_dnn_objdetect420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_dnn_superres420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_dpm420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_face420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_fuzzy420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_hfs420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_img_hash420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_line_descriptor420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_quality420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_reg420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_saliency420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_stereo420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_structured_light420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_superres420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_surface_matching420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_tracking420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_videostab420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_xfeatures2d420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_xobjdetect420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_xphoto420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_shape420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_highgui420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_datasets420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_plot420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_text420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_dnn420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_ml420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_phase_unwrapping420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_optflow420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_ximgproc420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_video420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_videoio420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_imgcodecs420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_objdetect420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_calib3d420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_features2d420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_flann420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_photo420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_imgproc420.dll.a
blog3.exe: C:/OpenCV_MinGW64/build/install/x64/mingw/lib/libopencv_core420.dll.a
blog3.exe: CMakeFiles/blog3.dir/linklibs.rsp
blog3.exe: CMakeFiles/blog3.dir/objects1.rsp
blog3.exe: CMakeFiles/blog3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\XX\CLionProjects\hci_project\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable blog3.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\blog3.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/blog3.dir/build: blog3.exe

.PHONY : CMakeFiles/blog3.dir/build

CMakeFiles/blog3.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\blog3.dir\cmake_clean.cmake
.PHONY : CMakeFiles/blog3.dir/clean

CMakeFiles/blog3.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\XX\CLionProjects\hci_project C:\Users\XX\CLionProjects\hci_project C:\Users\XX\CLionProjects\hci_project\cmake-build-debug C:\Users\XX\CLionProjects\hci_project\cmake-build-debug C:\Users\XX\CLionProjects\hci_project\cmake-build-debug\CMakeFiles\blog3.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/blog3.dir/depend

