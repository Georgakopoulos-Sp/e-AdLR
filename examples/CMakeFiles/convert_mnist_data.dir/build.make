# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

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

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/legolas/CNN_libs/caffe-master_last

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/legolas/CNN_libs/caffe-master_last

# Include any dependencies generated for this target.
include examples/CMakeFiles/convert_mnist_data.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/convert_mnist_data.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/convert_mnist_data.dir/flags.make

examples/CMakeFiles/convert_mnist_data.dir/mnist/convert_mnist_data.cpp.o: examples/CMakeFiles/convert_mnist_data.dir/flags.make
examples/CMakeFiles/convert_mnist_data.dir/mnist/convert_mnist_data.cpp.o: examples/mnist/convert_mnist_data.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/legolas/CNN_libs/caffe-master_last/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/convert_mnist_data.dir/mnist/convert_mnist_data.cpp.o"
	cd /home/legolas/CNN_libs/caffe-master_last/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/convert_mnist_data.dir/mnist/convert_mnist_data.cpp.o -c /home/legolas/CNN_libs/caffe-master_last/examples/mnist/convert_mnist_data.cpp

examples/CMakeFiles/convert_mnist_data.dir/mnist/convert_mnist_data.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/convert_mnist_data.dir/mnist/convert_mnist_data.cpp.i"
	cd /home/legolas/CNN_libs/caffe-master_last/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/legolas/CNN_libs/caffe-master_last/examples/mnist/convert_mnist_data.cpp > CMakeFiles/convert_mnist_data.dir/mnist/convert_mnist_data.cpp.i

examples/CMakeFiles/convert_mnist_data.dir/mnist/convert_mnist_data.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/convert_mnist_data.dir/mnist/convert_mnist_data.cpp.s"
	cd /home/legolas/CNN_libs/caffe-master_last/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/legolas/CNN_libs/caffe-master_last/examples/mnist/convert_mnist_data.cpp -o CMakeFiles/convert_mnist_data.dir/mnist/convert_mnist_data.cpp.s

# Object files for target convert_mnist_data
convert_mnist_data_OBJECTS = \
"CMakeFiles/convert_mnist_data.dir/mnist/convert_mnist_data.cpp.o"

# External object files for target convert_mnist_data
convert_mnist_data_EXTERNAL_OBJECTS =

examples/mnist/convert_mnist_data: examples/CMakeFiles/convert_mnist_data.dir/mnist/convert_mnist_data.cpp.o
examples/mnist/convert_mnist_data: examples/CMakeFiles/convert_mnist_data.dir/build.make
examples/mnist/convert_mnist_data: lib/libcaffe.so.1.0.0
examples/mnist/convert_mnist_data: lib/libcaffeproto.a
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/libboost_system.so
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/libboost_thread.so
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/libglog.so
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/libgflags.so
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/libprotobuf.so
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_cpp.so
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/libpthread.so
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/libsz.so
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/libz.so
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/libdl.so
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/libm.so
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl_cpp.so
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl.so
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/liblmdb.so
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/libleveldb.so
examples/mnist/convert_mnist_data: /usr/local/cuda/lib64/libcudart.so
examples/mnist/convert_mnist_data: /usr/local/cuda/lib64/libcurand.so
examples/mnist/convert_mnist_data: /usr/local/cuda/lib64/libcublas.so
examples/mnist/convert_mnist_data: /usr/local/cuda/lib64/libcudnn.so
examples/mnist/convert_mnist_data: /usr/local/lib/libopencv_highgui.so.3.4.3
examples/mnist/convert_mnist_data: /usr/local/lib/libopencv_videoio.so.3.4.3
examples/mnist/convert_mnist_data: /usr/local/lib/libopencv_imgcodecs.so.3.4.3
examples/mnist/convert_mnist_data: /usr/local/lib/libopencv_imgproc.so.3.4.3
examples/mnist/convert_mnist_data: /usr/local/lib/libopencv_core.so.3.4.3
examples/mnist/convert_mnist_data: /usr/local/lib/libopencv_cudev.so.3.4.3
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/liblapack.so
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/libcblas.so
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/libatlas.so
examples/mnist/convert_mnist_data: /usr/lib/x86_64-linux-gnu/libboost_python.so
examples/mnist/convert_mnist_data: examples/CMakeFiles/convert_mnist_data.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/legolas/CNN_libs/caffe-master_last/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mnist/convert_mnist_data"
	cd /home/legolas/CNN_libs/caffe-master_last/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/convert_mnist_data.dir/link.txt --verbose=$(VERBOSE)
	cd /home/legolas/CNN_libs/caffe-master_last/examples && ln -sf /home/legolas/CNN_libs/caffe-master_last/examples/mnist/convert_mnist_data /home/legolas/CNN_libs/caffe-master_last/examples/mnist/convert_mnist_data.bin

# Rule to build all files generated by this target.
examples/CMakeFiles/convert_mnist_data.dir/build: examples/mnist/convert_mnist_data

.PHONY : examples/CMakeFiles/convert_mnist_data.dir/build

examples/CMakeFiles/convert_mnist_data.dir/clean:
	cd /home/legolas/CNN_libs/caffe-master_last/examples && $(CMAKE_COMMAND) -P CMakeFiles/convert_mnist_data.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/convert_mnist_data.dir/clean

examples/CMakeFiles/convert_mnist_data.dir/depend:
	cd /home/legolas/CNN_libs/caffe-master_last && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/legolas/CNN_libs/caffe-master_last /home/legolas/CNN_libs/caffe-master_last/examples /home/legolas/CNN_libs/caffe-master_last /home/legolas/CNN_libs/caffe-master_last/examples /home/legolas/CNN_libs/caffe-master_last/examples/CMakeFiles/convert_mnist_data.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/convert_mnist_data.dir/depend

