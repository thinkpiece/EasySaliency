EasySaliency
===

This is an implementation of the saliency model of the visual attention. The implementation mostly follows Itti’s TPAMI1998, except the winner-take-all neural network. (*will be added next time*) See the paper below for the implementation details.

> L. Itti, C. Koch, and E. Niebur, "A model of saliency-based visual attention for rapid scene analysis," Pattern Analysis and Machine Intelligence, IEEE Transactions on, vol. 20, no. 11, pp. 1254–1259, 1998.

To run the model, CMake & OpenCV (>2.0) is required to run this code. Since the OpenCV cannot be installed without CMake, all you need is a working OpenCV library.

Simple demonstration is written in **demo** subdirectory. Follow the steps below.

1. Make the compile directory and generate Makefile using CMake. There are several options for shared/static libraries.

	```
	$ mkdir test 
	$ cd test
	$ cmake ..
	```
	
2. Make, and run the demo application. Image (larger than 640x480) is required.

	```
	$ cd demo
	 $ ./saliency_demo
	```

Files organization
===

The included files are:

- **src/saliency.h, .cc**: The saliency model implementation.
- **src/normalizer.h, .cc**: The normalization operator implementation. There are two types for normalization, one is Local Max based, and another is Iterative Method based. For the details, see the paper below.

	> L. Itti and C. Koch, "Comparison of feature combination strategies for saliency-based visual attention systems," Electronic Imaging'99, pp. 473–482, May 1999.

- **src/scalespace.h, .cc**: The scale-space implementation.
- **src/imgdump.h, .cc**: To export the internal images for executing the saliency map.
- **demo/demo.cc**: A simple demonstration.

