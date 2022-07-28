# semantic-robot-multitask
We will start with Reinplementing CLIPort, but general theme is robot multitasking with language


# Install requirements
If you intend to use robosuite or other environments requiring mujoco, install that first.

**NOTE:** since Python 3.7 > you can install mujoco with `pip install mujoco` but don't install it that way! other packages might require older version and it is recommended to install
(mujoco 2.1)[https://github.com/deepmind/mujoco/releases/tag/2.1.0]. 
### Mujoco (mujoco-py) requirements
on Linux (Ubuntu) PC, please first install the following packages to build `mujoco-py`

```
sudo apt install curl git libgl1-mesa-dev libgl1-mesa-glx libglew-dev \
         libosmesa6-dev software-properties-common net-tools unzip vim \
         virtualenv wget xpra xserver-xorg-dev libglfw3-dev patchelf
```
Then download the compressed file (mujoco 210)[https://github.com/deepmind/mujoco/releases/tag/2.1.0] and extract it in the folder `~/.mujoco/mujoco210`.

Now you should be able to install `robosuite` and `dm-control` if you need those.

## PyBullet and benchmarks
TBD


## Install torch & Clip

Install Pytorch and torchvision with cuda support for your system, or simply 
``` pip install -r requirements.txt ``` 
and CLIP + pytorch + torchvision (and tqdm) will be installed.

