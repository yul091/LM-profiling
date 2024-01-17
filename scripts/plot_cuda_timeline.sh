#!/bin/bash
stages=3
ngpus=8
microbs=32

base_dir=prof
name=${pipeline}_transformer-${nlayers}_${stages}stages_${ngpus}gpus_microbs${microbs}
main_event_text=call_pipeline

sqlite_path=${name}.sqlite
pickle_path_timeline=${base_dir}/${name}_timeline.pickle

echo parse $sqlite_path
python parse_nvtx_events.py \
    $sqlite_path \
    --pickle_path_timeline $pickle_path_timeline \
    --ignore_first_event \
    --main_event_indices '2,4,5' \
    --event_keywords call_forward,call_backward,precondition,reduce,gather,sync \
    --main_event_text $main_event_text

# rm -f $sqlite_path
nsys_path=${base_dir}/$(basename ${sqlite_path} | cut -f 1 -d '.' ).nsys-rep
# rm -f $nsys_path

pickle_paths=""
for pickle_path in $(find ${base_dir} -type f -name "${name}_timeline.pickle" | sort )
do
    pickle_paths+="${pickle_path},"
done
fig_path=${base_dir}/${name}.pdf
echo creating ${fig_path} ...
python plot_cuda_timeline.py \
    $pickle_paths \
    --fig_path $fig_path \
    --title $name \
    --num_replicas 1 \

#imgcat $fig_path
