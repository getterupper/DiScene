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

# For InternImage
cd models/backbones/ops_dcnv3
python setup.py build install
```

**f. Install DepthAnythingV2.**
```shell
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cd Depth-Anything-V2
pip install -r requirements.txt
```
Go to Depth-Anything-V2/metric_depth/depth_anything_v2/dpt.py and change the function infer_image in the class DepthAnythingV2 as follows:
```python
def infer_image(self, image, h_, w_, input_size=518):
    depth = self.forward(image)
    depth = F.interpolate(depth[:, None], (h_, w_), mode="bilinear", align_corners=True)[0, 0]
    return depth
```
And download the fine-tuned model from [HERE](https://huggingface.co/YkiWu/EmbodiedOcc).

**g. Install Metric3Dv2 for training.**
```shell
git clone https://github.com/YvanYin/Metric3D.git
cd Metric3D
pip install -r requirements_v2.txt
```

**h. modify paths**

Finally, remember to modify these lines in [student head](../models/heads/discene_student_head.py), [teacher head](../models/heads/discene_teacher_head.py) and [distill head](../models/heads/discene_distill_head.py) to your own paths if used.
```python
# NOTE: modify below to /your/path/to/DiScene
sys.path.append('/path/to/DiScene')
sys.path.append('/path/to/DiScene/depth_anything/metric_depth')
sys.path.append('/path/to/DiScene/Depth-Anything-V2/metric_depth')
sys.path.append('/path/to/DiScene/Metric3D')
```
DepthAnythingV1:
```python
if pretrained_depth_model == 'DepthAnythingV1':
    # NOTE: modify below to /your/path/to/DiScene
    overrite = {"pretrained_resource": "local::/path/to/DiScene/checkpoints/depth_anything_metric_depth_indoor.pt"}
```
DepthAnythingV2:
```python
self.depth_model = DepthAnythingV2(**{**model_configs['vitb'], 'max_depth':20})
# NOTE: modify below to /your/path/to/DiScene
checkpoint = torch.load('/path/to/DiScene/checkpoints/finetune_scannet_depthanythingv2.pth', map_location='cpu')['model']
```
Metric3Dv2-Small:
```python
# NOTE: modify below to /your/path/to/DiScene
cfg = Config.fromfile('/path/to/DiScene/Metric3D/mono/configs/HourglassDecoder/vit.raft5.small.py')
cfg.load_from = '/path/to/.cache/torch/hub/checkpoints/metric_depth_vit_small_800k.pth'
```
Metric3Dv2-Giant:
```python
# NOTE: modify below to /your/path/to/DiScene
cfg = Config.fromfile('/path/to/DiScene/Metric3D/mono/configs/HourglassDecoder/vit.raft5.giant2.py')
cfg.load_from = '/path/to/.cache/torch/hub/checkpoints/metric_depth_vit_giant2_800k.pth'
```