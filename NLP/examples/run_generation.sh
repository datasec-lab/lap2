export TRANSFORMERS_CACHE=cache
data_dir=table2text/prefix-tuning  # download dataset before running this file.
save_path=../../results
lap2_config_dir=lap2_configs
task_name=e2e # dart
gpu_id=0

eps=0.5
lap2_config=${lap2_config_dir}/eps_05.yaml

lap2=True # False
modelname=distilgpt2 # gpt2 gpt2-medium gpt2-large
clip=0.5


noise_type=gaussian
if [[ "${lap2,,}" == "true" ]]; then
    noise_type=lap2
fi

clipping_fn=Abadi
clipping_mode=ghost
clipping_style=all-layer
bias_only=no
non_private=no
learning_rate=0.002
batch_size=792
attention_only=no
static_lm_head=static_lm_head
static_embedding=no
physical_batch_size=100

tag=${task_name}_${modelname}_${clipping_mode}_eps${eps}_C${clip}
if [[ "${lap2,,}" == "true" ]]; then
    tag="lap2_${tag}"
fi

save_dir=${save_path}/${task_name}/${noise_type}/${tag}
mkdir -p ${save_dir}

bash table2text/run_gen.sh ${data_dir} ${save_dir} ${task_name} ${modelname} \
    ${eps} ${clipping_fn} ${clipping_mode} ${clipping_style} ${bias_only} \
    ${non_private} ${physical_batch_size} ${learning_rate} ${batch_size} \
    ${attention_only} ${static_lm_head} ${static_embedding} ${gpu_id} ${clip} ${lap2} ${lap2_config}
