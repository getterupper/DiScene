# Installation instructions

**a. Create a conda virtual environment and activate it.**

```shell
conda create -n discene python=3.8.19
conda activate discene
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**

```shell
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --index-url https://download.pytorch.org/whl/cu113
```

**c. Install mmcv-full.**

```shell
pip install openmim
mim install mmcv-full==1.6.0
```

**d. Install other dependencies.**

```shell
mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0
mim install mmdet3d==1.0.0rc6
pip install -r requirements.txt
```

**e. Compile CUDA extensions.**

```shell
cd models/csrc
python setup.py build_ext --inplace
```
