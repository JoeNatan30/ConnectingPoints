git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git

cd openpose/3rdparty/
git clone https://github.com/CMU-Perceptual-Computing-Lab/caffe.git
git clone https://github.com/pybind/pybind11

cd ..

mkdir build
cd build

sudo apt-get install protobuf-compiler libprotobuf-dev libgoogle-glog-dev libblkid-dev e2fslibs-dev libboost-all-dev libaudit-dev libcanberra-gtk-module libcanberra-gtk3-module
sudo apt-get install libhdf5-dev
sudo apt install -y libatlas3-base libatlas-base-dev


# Remeber to have CMake version to 3.12.2 or later to avoid CUDA_cublas_device_LIBRARY error
# import openpose in python works for python 3.8
cmake -S .. -B . -DBUILD_PYTHON=ON -DUSE_CUDNN=OFF -DCUDA_ARCH=Manual -DCUDA_ARCH_BIN="60 61 62" -DCUDA_ARCH_PTX="61" -DPYTHON_EXECUTABLE=~/miniconda3/bin/python3.8 -DPYTHON_LIBRARY=~/miniconda3/lib/libpython3.8.so.

make -j`nproc`

sudo make install

cd python/openpose

sudo make install

#./build/examples/openpose/openpose.bin --video ../../../datasets/AEC/Videos/SEGMENTED_SIGN/ira_alegria/ahí_328.mp4 --display 0 --render_pose 0 --write_json output/
