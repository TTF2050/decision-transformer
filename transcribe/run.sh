export PYTHONUNBUFFERED="true"

python experiment.py --env hopper --dataset medium  | tee "hopper-medium.log"
# --max_iters 1 --num_steps_per_iter 1


# for GYM_ENV in "hopper" "halfcheetah" "walker2d"; do
#     for DATASET in "medium" "medium-replay" "expert"; do
#         python experiment.py --env ${GYM_ENV} --dataset ${DATASET} --model_type dt | tee "${GYM_ENV}-${DATASET}.log"
#     done
# done