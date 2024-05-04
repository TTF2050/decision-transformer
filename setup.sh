## for reference only
exit 1

## initial setup on host/login node
cd ~
mkdir dectransform
cd dectransform
#read-only clone and checkout
git clone https://github.com/TTF2050/decision-transformer.git 
cd decision-transformer
git checkout oscar
cd ../..

#build image
singularity build dectransform.simg docker://ttf2050/dectransform:0.0.1
#start image
singularity run --nv --bind dectransform:${HOME}/dectransform dectransform.simg
#put the env configuration script in the right place
cd dectransform
cp decision-transformer/bootstrap_env.sh .
./bootstrap_env.sh
source .env/bin/activate
cd decision-transformer/gym/data
python download_d4rl_datasets.py
# exit image shell ctrl+d


## do stuff
interact -q gpu -g 2
singularity run --nv --bind dectransform:${HOME}/dectransform dectransform.simg
cd dectransform/
source .env/bin/activate
cd decision-transformer/gym
./run.sh
