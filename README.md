# Egocentric 3D Hand Pose Estimation
- This repository reproduces the [WildHands](https://ap229997.github.io/projects/hands/) method and evaluates the impact of different backbone architectures in order to analyze how backbone choice affects performance and generalization in egocentric 3D hand pose estimation.
  - The default ResNet-50 backbone is swapped with:
    - [MoblieNet-V3-Large](https://arxiv.org/abs/1905.02244)
    - [MoblieVIT-S](https://arxiv.org/abs/2110.02178)
    - [ResNet-101](https://arxiv.org/abs/1512.03385)
    - [ConvNeXt-L](https://arxiv.org/abs/2201.03545)
    - [DF-Mamba](https://arxiv.org/abs/2512.02727)
  - Mirroring the training/evaluation methodology from WildHands we:
    - Use an egocentric split of the [ARCTIC](https://github.com/zc-alexfan/arctic/blob/master/docs/setup.md), [AssemblyHands](https://assemblyhands.github.io/), [Epic-Kitchens](https://epic-kitchens.github.io/VISOR/), [Ego4D](https://ego4d-data.org/docs/start-here/) for training.
    - Conduct zero-shot evaluation on the [H2O](https://taeinkwon.com/projects/h2o/), [AssemblyHands](https://assemblyhands.github.io/), [EPIC-HandKps](https://drive.google.com/drive/folders/18hvFlt3rBl2vjSGsFh1kRWPK_mjLCAZc?usp=sharing) and [Ego-Exo4D](https://ego4d-data.org/docs/start-here/) datasets

# Setup
## Prerequisites

- **NVIDIA GPU**
- **Python 3.10** 
- **Git**
- **Anaconda Distribution**
## 1. Clone the repo

```shell
git clone https://github.com/BJKin/Egocentric-3D-Hand-Pose-Estimation.git
cd Egocentric-3D-Hand-Pose-Estimation
```
## 2. Set up the environment (Windows/PowerShell)

This project targets the following stack:

| Component         | Version |
|-------------------|---------|
| Python            | 3.10    |
| CUDA Toolkit      | 11.6    |
| PyTorch           | 1.13.0  |
| PyTorch3D         | 0.7.3   |
| PyTorch Lightning | 2.0.0   |
| aitviewer         | 1.8.0   |

All commands below assume a PowerShell session with conda available. If running `conda` errors out, you'll need to wire it up once with `conda init powershell` from an Anaconda Prompt and reopen PowerShell.

---

### Step 1 - CUDA Toolkit 11.6

Grab the Windows installer from NVIDIA's [CUDA Toolkit 11.6 download page](https://developer.nvidia.com/cuda-11-6-0-download-archive) and run it. Once it finishes, confirm the toolchain is wired up correctly:

```powershell
nvcc --version       # expect: Cuda compilation tools, release 11.6
Get-Command nvcc     # shows which nvcc.exe is being picked up
```

A common pitfall is having multiple CUDA versions installed and the wrong one shadowing 11.6 on PATH. Force the correct one for this session:

```powershell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6"
$env:PATH      = "$env:CUDA_HOME\bin;$env:CUDA_HOME\libnvvp;$env:PATH"
```

To make that stick across reboots, write it to the user environment with `setx` (you'll need to launch a new shell afterward for it to take effect):

```powershell
setx CUDA_HOME "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6"
```

---

### Step 2 - Conda environment

Spin up a fresh Python 3.10 environment:

```powershell
$ENV_NAME = "Ego3D_env"
conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME
```
If PowerShell complains that `conda` is not a valid command then first do the following, then try again:

Open the "Anaconda PowerShell Prompt"
``` powershell
conda init powershell
```
Then close and reopen your regular PowerShell.

---

### Step 3 - PyTorch and PyTorch3D

PyTorch installs cleanly from conda. PyTorch3D does not, the `pytorch3d` conda
channel has no Windows builds for 0.7.3, and pip-installing from source on Windows
requires a specific toolchain. The recipe below works.

#### 3a. Install PyTorch

```powershell
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
```

#### 3b. Prerequisites for the PyTorch3D source build

Three things need to be in place before pip can build pytorch3d:

- **Visual Studio 2022 Build Tools** with the "Desktop development with C++"
  workload. Download from
  <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022>.

- **An older MSVC toolset (v14.38).** CUDA 11.6's `nvcc` rejects newer MSVC
  versions (14.40+) with a hard `unsupported Microsoft Visual Studio version`
  error. The `-allow-unsupported-compiler` flag that's commonly suggested does
  not work for this stack. The reliable fix is to install an older toolset
  alongside the default one:
  1. Open the Visual Studio Installer (search "Visual Studio Installer" in Start).
  2. Click Modify on "Visual Studio Build Tools 2022."
  3. Switch to the **Individual components** tab and search `14.38`.
  4. Check `MSVC v143 - VS 2022 C++ x64/x86 build tools (v14.38-17.8)` and click Modify.

- **CUB headers v1.10.0.** Download from
  <https://github.com/NVIDIA/cub/releases/tag/1.10.0> and unzip to
  `C:\cub-1.10.0`. Verify with `dir C:\cub-1.10.0\CMakeLists.txt`.

#### 3c. Build pytorch3d from source

Open **x64 Native Tools Command Prompt for VS 2022** from the Start menu — not
regular PowerShell, and not the plain "Developer Command Prompt" (that one
initializes a 32-bit environment, which won't work). The prompt's header should
read `Environment initialized for: 'x64'`.

In that prompt:

```bat
conda activate Ego3D_env
set VCToolsVersion=14.38.33130
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6
set CUB_HOME=C:\cub-1.10.0
set DISTUTILS_USE_SDK=1
pip install "setuptools<81"
pip install ninja
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.3"
```


#### 3d. Verify

Back in regular PowerShell:

```powershell
conda activate Ego3D_env
python -c "import pytorch3d; print(pytorch3d.__version__)"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Should print `0.7.3`, then `True` and the GPU name.

---

### Step 4 - Remaining Python packages

The default install path fights with PyTorch 1.13. Both `pytorch_lightning` and
`torchmetrics` will try to drag torch forward to 2.x, which breaks the
pytorch3d build from Step 3. A pip constraints file pins the versions that
matter so the resolver can't upgrade them.

Then install in this order:

```powershell
conda install pytorch=1.13.0 torchvision=0.14.0 pytorch-cuda=11.6 -c pytorch -c nvidia -y --force-reinstall
pip install --no-build-isolation chumpy
pip install -c constraints.txt -r requirements.txt
```

Verify nothing got clobbered and CUDA is wired up::

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
python -c "import pytorch3d; print(pytorch3d.__version__)"
```

Should print `1.13.0 True 11.6` and `0.7.3`. If `cuda.is_available()` returns
False, you've got a CPU-only torch build, run the conda command again.


---

### Step 5 - smplx patch

The default `smplx` build returns 16 joints, but the model expects 21. The fix is a one-line uncomment in `body_models.py` inside the installed package.

Locate the file and open it for editing:

```powershell
$smplx_file = python -c "import smplx, os; print(os.path.join(os.path.dirname(smplx.__file__), 'body_models.py'))"
notepad $smplx_file       # or: code $smplx_file
```

Jump to line 1681 and uncomment:

```python
joints = self.vertex_joint_selector(vertices, joints)
```

Save and close. Setup is done.

## 3. Download the data
| Dataset           | Role                        | Access                                                          |
|-------------------|-----------------------------|-----------------------------------------------------------------|
| ARCTIC            | Training                    | [arctic.is.tue.mpg.de](https://arctic.is.tue.mpg.de/)           |
| AssemblyHands     | Training + zero-shot eval   | [assemblyhands.github.io](https://assemblyhands.github.io/)     |
|VISOR (Epic-Kitchens) | Training                | [epic-kitchens.github.io/VISOR](https://epic-kitchens.github.io/VISOR/) |
| Ego4D             | Training                    | [ego4d-data.org](https://ego4d-data.org/)                       |
| H2O               | Zero-shot eval              | [taeinkwon.com/projects/h2o](https://taeinkwon.com/projects/h2o/) |
| Epic-HandKps      | Zero-shot eval              | (annotations included in the preprocessed bundle below)         |
| Ego-Exo4D         | Zero-shot eval              | [ego-exo4d-data.org](https://ego-exo4d-data.org/)               |

Each dataset has its own access flow, most require a license agreement or signed form. 

### Step 1- Preprocessed Labels (WildHands)

The WildHands authors provide preprocessed segmentation masks, grasp labels, and pickle files that the dataloaders expect (annotations only). 

- Main bundle: [Google Drive](https://drive.google.com/drive/folders/1rtrhOoEVUsJJEGYJLC5y8ZJo6tDzpdLd)
- EPIC-HandKps eval annotations: [Google Drive](https://drive.google.com/drive/folders/1ggF9FKrkAIdv6RAB-xr1Tolh_Nt6lEJb) 
  - Download `hands_5000.pkl`, place under `data/epic_hands/`

### Step 2 - MANO Hand Model

The MANO model files drive the 3D hand mesh regression head. Register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/), accept the license, and download the model files (`MANO_RIGHT.pkl`, `MANO_LEFT.pkl`). Place them in a top-level `mano/` directory.

### Step 3 - Pretrained Weights for Initialization

WildHands is initialized from an ArcticNet checkpoint pretrained on the allocentric split of ARCTIC. Grab the checkpoint from the [ARCTIC data page](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md) and stash it under `data/checkpoints/`.

### Expected Directory Layout

After everything is downloaded, the repo should look like this:

```
Egocentric-3D-Hand-Pose-Estimation/
├── data/
│   ├── arctic/            # raw ARCTIC frames
│   ├── assembly/          # raw AssemblyHands frames
│   ├── h2o/               # raw H2O frames
│   ├── visor/             # raw Epic-Kitchens VISOR frames
│   ├── ego4d/             # raw Ego4D + Ego-Exo4D frames
│   ├── epic_hands/        # preprocessed pkls
│   ├── ego4d_hands/       # preprocessed pkls for Ego4D/Ego-Exo4D
│   └── checkpoints/       # ArcticNet pretrained weights
├── mano/                  # MANO_RIGHT.pkl, MANO_LEFT.pkl
├── src/                   # model code
├── scripts/               # training + eval entrypoints
├── requirements.txt
└── README.md
```


## 4. Set Environment Variables

The training and evaluation scripts read paths from `MANO_DIR` and `DATA_DIR`. Set them once per shell session:

```powershell
$env:MANO_DIR = "$PWD\mano"
$env:DATA_DIR = "$PWD\data"
```

To make them stick across reboots (open a new PowerShell window afterward for the change to apply):

```powershell
setx MANO_DIR "$PWD\mano"
setx DATA_DIR "$PWD\data"
```

# Acknowledgements

Setup procedure, dataloader structure, and training framework adapted from [WildHands (Prakash et al., ECCV 2024)](https://github.com/ap229997/hands). MANO model from [Romero et al., 2017](https://mano.is.tue.mpg.de/). Refer to each upstream dataset and codebase for its citation and license.