
# Run the demo/ctr/widedeep

```bash

export MY_S3_BUCKET='s3a://sagemaker-us-west-2-452145973879/datasets/'
envsubst < fg.yaml > fg.yaml.dev 

envsubst < match_dataset_cf.yaml > match_dataset.yaml.dev 
nohup python match_dataset_cf.py --conf match_dataset.yaml.dev --verbose > log_match.out 2>&1 &


envsubst < match_dataset_negsample_10.yaml > match_dataset_negsample_10.yaml.dev 
nohup python match_dataset_negsample.py --conf match_dataset_negsample_10.yaml.dev --verbose > log_match_neg.out 2>&1 &

envsubst < rank_dataset.yaml > rank.yaml.dev

nohup python rank_dataset.py --conf rank.yaml.dev --verbose > log_rank.out 2>&1 &


cd ctr/widedeep/conf
envsubst < widedeep_ml_1m.yaml > widedeep_ml_1m.yaml.dev


cd MetaSpore/demo/ctr/widedeep
python widedeep.py --conf conf/widedeep_ml_1m.yaml.dev

```

# Recompile metaspore

## Step 1: Install the previous metaspore

### 离线安装包
我们提供了预编译的 Python 安装包，可以通过 pip 安装：
```bash
pip install metaspore
```
支持 Python 的最低版本为 3.8。

运行 MetaSpore 离线训练，还需要 PySpark 和 PyTorch。可以通过 `pip` 命令进行安装：
这两个依赖没有作为 metaspore wheel 的默认依赖，这样方便用户选择需要的版本。

Spark 官方打包的 PySpark，没有包含 hadoop-cloud 的 jar 包，无法访问 S3 等云存储。我们提供了一个打包好 S3 客户端的 [PySpark 安装包](https://ks3-cn-beijing.ksyuncs.com/dmetasoul-bucket/releases/spark/pyspark-3.1.2.265f9ad4ee-py2.py3-none-any.whl)，可以从这里下载后安装：
```bash
pip install https://ks3-cn-beijing.ksyuncs.com/dmetasoul-bucket/releases/spark/pyspark-3.1.2.265f9ad4ee-py2.py3-none-any.whl
```

```bash
pip install torch==1.11.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```


## Step 2:  Recompile the new metaspore

```bash
sudo apt install build-essential manpages-dev software-properties-common curl zip unzip tar pkg-config bison flex python3-dev

sudo add-apt-repository ppa:ubuntu-toolchain-r/test

sudo apt-get update

sudo apt install gcc-11 g++-11

# optional steps if you have multiple versions of gcc/g++
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9 --slave /usr/bin/gcov gcov /usr/bin/gcov-9 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-9 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-9

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 --slave /usr/bin/g++ g++ /usr/bin/g++-11 --slave /usr/bin/gcov gcov /usr/bin/gcov-11 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-11 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-11

# install latest cmake
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
 
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" 

sudo apt update
sudo apt install cmake

# MetaSpore uses vcpkg to manage thirdparty c++ dependencies
git clone https://github.com/Microsoft/vcpkg.git ~/.vcpkg
~/.vcpkg/bootstrap-vcpkg.sh

cd ~/.vcpkg/triplets

# create a ~/.vcpkg/triplets/x64-linux-custom.cmake with the following contents
# This is the issue which leads to undefined symbol or unreferenced symbol during link, D_GLIBCXX_USE_CXX11_ABI=0 should be used for all packages installed using vcpkg, since CMakeFiles.txt also used this.

cat <<EOF > ~/.vcpkg/triplets/x64-linux-custom.cmake
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
set(VCPKG_C_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
set(VCPKG_CMAKE_SYSTEM_NAME Linux)
EOF

git clone https://github.com/meta-soul/MetaSpore.git
cd MetaSpore

# Download MNIST dataset from s3
mkdir cpp/tests/data/MNIST
aws s3 sync s3://sagemaker-us-west-2-452145973879/datasets/MNIST/ cpp/tests/data/MNIST

mkdir build && cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=~/.vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-linux-custom -DBUILD_TRAIN_PKG=ON -DBUILD_SERVING_BIN=ON -DENABLE_TESTS=ON

make -j8

# Then install  
pip install metaspore-1.2.0-cp38-cp38-linux_x86_64.whl

# Then use the following code to test
python widedeep.py --conf conf/widedeep_ml_1m.yaml.dev
```