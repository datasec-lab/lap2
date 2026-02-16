export TRANSFORMERS_CACHE=cache
data_dir=data/  # download dataset before running this file.
save_path=../../results
lap2_config_dir=lap2_configs
task_name=sst-2 # qnli
gpu_id=0

eps=1
lap2=True # False
modelname=bert-base-uncased # bert-large-uncased roberta-base roberta-large
clip=0.4

noise_type=gaussian
if [[ "${lap2,,}" == "true" ]]; then
    noise_type=lap2
fi

seed=42
clipping_fn=Abadi
clipping_mode=ghost
clipping_style=all-layer
bias_only=no
non_private=no
batch_size=1024
physical_batch_size=8

tag=${task_name}_${modelname}_${clipping_mode}_eps${eps}_C${clip}
if [[ "${lap2,,}" == "true" ]]; then
    tag="lap2_${tag}"
fi

save_dir=${save_path}/${task_name}/${noise_type}/${tag}
mkdir -p ${save_dir}

lap2_config=${lap2_config_dir}/test.yaml

CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m text_classification.run_wrapper \
    --seed ${seed} \
    --output_dir ${save_dir} \
    --task_name ${task_name} \
    --few_shot_type finetune \
    --eval_steps 100 \
    --target_epsilon ${eps} \
    --clipping_mode ${clipping_mode} \
    --clipping_fn ${clipping_fn} \
    --clipping_style ${clipping_style} \
    --bias_only ${bias_only} \
    --non_private ${non_private} \
    --model_name_or_path ${modelname} \
    --num_train_epochs 3 \
    --per_example_max_grad_norm ${clip} \
    --physical_batch_size ${physical_batch_size} \
    --batch_size ${batch_size} \
    --plrv ${plrv} \
    --plrv_config ${lap2_config}
    
wait
