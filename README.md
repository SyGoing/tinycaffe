轻量级caffe前向推理框架

一、简单介绍
本项目为对happynear版本的caffe进行精简化:
1、去除冗余依赖boost、hdf5、lmdb、levedb等，简化以后caffe前向框架更加小型化，并且保持了原生caffe调用习惯（相比于mini-caffe）
2、去除训练用到的loss计算以及solver源码
3、调整修改内存管理机制（去除boost）

小赞：
	相比于mini-caffe，该精简化版本的caffe更加原生态，函数调用和原版的caffe没有区别，同时也支持mkl编译，多GPU模式编译（nccl)
	提供的cmake可以在ubuntu下编译，同时也可以在windows下编译使用。
<<<<<<< HEAD
二、编译
	1、windows下编译，blob.hpp、io.hpp、net.hpp、common.hpp、layer.hpp、caffe.pb.h的文件开始加入：
=======
	
	
二、编译
    目前仅仅只是在windows下呢，我提交了编译的sln工程和编译好的库（libwrap文件）,这个版本的caffe支持cuda10.1和cudnn7.5.1。
	1、windows下编译，blob.hpp、io.hpp、net.hpp、common.hpp、layer.hpp、caffe.pb.h的文件开始加入：
		1）、编译时：
				#define LIBCAFFE __declspec(dllexport)
		2）、调用时：
				#ifdef CAFFE_EXPORTS
				#define LIBCAFFE __declspec(dllexport)
				#else
				#define LIBCAFFE __declspec(dllimport)
				#endif
>>>>>>> first
	   
	   三方依赖下载地址（百度网盘): https://pan.baidu.com/s/15x-5n9hb5UAkUlEm0uvy-A   
	   (password: 6b8d（3rdparty(BaiduYun):https://pan.baidu.com/s/15x-5n9hb5UAkUlEm0uvy-A   password: 6b8d)
	   
	   三方依赖下载好以后，解压覆盖文件3rdparty,3rd文件结构如下：
	   3rdparty-
		---bins
		---include
		---libs
		 
<<<<<<< HEAD
	   	*************注意：
		1）、编译时：
				#define LIBCAFFE __declspec(dllexport)
		2）、调用时：
				#ifdef CAFFE_EXPORTS
				#define LIBCAFFE __declspec(dllexport)
				#else
				#define LIBCAFFE __declspec(dllimport)
				#endif
			
	    这是一个坑，一开始编译时加入第二种宏定义方式，总是报错。最后不得已使用这种方式。ubuntu下不存在这些问题！
		
		
		********我已经生成了好一份caffe.pb.h和caffe.pb.cc，对应的protobuf直接使用3rdpaty的protobuf库
=======
		********我已经生成了好一份caffe.pb.h和caffe.pb.cc，对应的protobuf直接使用3rdpaty的protobuf(3.0.0)库
>>>>>>> first
		
   		
       如果需要添加新的层，需要重新生成caffe.pb.h和caffe.pb.cc，protobuf库自由选择，但是版本一定要一致。
       首先删除3rdparty中的google文件夹
       protobuf可以自己下载最新的protobuf到3rdparty/src，
		
		cd 3rdparty/src
		git clone https://github.com/protocolbuffers/protobuf.git
		cd protobuf/cmake
		mkdir build
		cd build
		cmake .. -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -G "Visual Studio 14 2015 Win64"
		or
        cmake .. -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -G "Visual Studio 12 2013 Win64"
		
	   Use protobuf.sln to compile Debug and Release version.
	   With these two libraries, we can compile Mini-Caffe now. Copy protobuf's include headers and libraries. Generate caffe.pb.h and caffe.pb.cc.
		
	    copydeps.bat
        generatepb.bat
        mkdir build
        
		自己按照文件结构新建控制台应用程序加入代码，并配置依赖的三方库
		


		
<<<<<<< HEAD
		
	2、ubuntu16.04 编译命令：
		 sudo apt install libopenblas-dev libprotobuf-dev protobuf-compiler
		 ./generatepb.sh
		 mkdir build
		 cd build
		 cmake .. -DCMAKE_BUILD_TYPE=Release
		 make -j4
=======
>>>>>>> first
