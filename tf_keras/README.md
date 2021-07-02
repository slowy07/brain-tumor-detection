## tensorflow installation
### tensorflow packages
 - ```tensorflow``` latest stable with CPU and [GPU Support](https://www.tensorflow.org/install/gpu) (ubuntu and windows)
 - ```tf-nightly``` priview build (unstable). ubuntu and windows include [GPU Support](https://www.tensorflow.org/install/gpu)

### system requirements 
 - [Python](https://www.python.org/) 3.6 - 3.9
    - Python 3.9 support requires tensorflow 2.5 or later
    - python 3.8 support requires tensorflow 2.2 or later
 - pip 19.0 or later (requires ```manylinux2010``` support)
 - ubuntu 16.04 or later (46bit)
 - macOS 10.12.6 (sierra) or later (64bit) (no GPU Support)
    - macOS requires pip 20.3 or later
 - windows 7 or later(64 bit)
    - [Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019](https://support.microsoft.com/help/2977003/the-latest-supported-visual-c-downloads)
 - [GPU Support](https://www.tensorflow.org/install/gpu) requires a CUDA®-enabled card (Ubuntu and Windows)

## installation
install tensorflow 
### install python dev env
```bash
python3 --version
pip3 --version
```
if these package already installed, skip to next step
```bash
sudo apt-get update
```
```bash
sudo apt-get install python3-dev python3-pip python3-venv
```
### donwload and install tensorflow pip package
choose of the following tensorflow packages to install from [pypi](https://pypi.org/project/tensorflow/)
 - ```tensorflow``` latest stable release CPU and [gpu support](https://www.tensorflow.org/install/gpu)
 - ```tf-nightly``` preview build (unstable). ubuntu and windows include [gpu support](https://www.tensorflow.org/install/gpu)
 - ```tensorflow==1.25``` the final version of tensorflow 1.x
system install 
```bash
pip3 install --user --upgrade tensorflow
```
virtual env install
```bash
pip3 install --upgrade tensorflow
```
verify the install
```bash
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```