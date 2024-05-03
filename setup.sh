#for reference only
exit 1

singularity build dectransform.simg docker://ttf2050/dectransform:0.0.1
interact -q gpu -g 2
singularity run --nv --bind dectransform:${HOME}/dectransform dectransform.simg
