#!/bin/bash


python -m venv --system-site-packages .env \
    && . .env/bin/activate \
    && pip install gymnasium torch "tensorflow==2.15" carla wandb "cython<3" mujoco \
     transformers atari-py opencv-python blosc dopamine-rl dopaminekit \
     tensorflow-estimator  \
     ray[all]
     #git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl


git clone https://github.com/TTF2050/d4rl.git && cd d4rl && git checkout update-to-gymnasium  && pip install -e .

VIRTUAL_ENV /home/user/.env
PATH /home/user/.env/bin:$PATH
python -c "import d4rl"

git clone https://github.com/karpathy/minGPT.git && cd minGPT && pip install -e .


curl -O https://www.atarimania.com/roms/Atari-2600-VCS-ROM-Collection.zip 
unzip Atari-2600-VCS-ROM-Collection.zip -d Atari-2600-VCS-ROM-Collection || true
python -m atari_py.import_roms Atari-2600-VCS-ROM-Collection 