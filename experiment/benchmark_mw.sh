# # usage: bash benchmark.sh <GPU_ID> <PATH_TO_BENCHMARK.PY>
# do 
#     CUDA_VISIBLE_DEVICES=$1 accelerate launch benchmark_mw.py --env_name $task --n_exps 25 --ckpt_dir "../ckpts/metaworld" --milestone 24 --result_root "../results/results_AVDC_mw"
# done

# for task in "door-open-v2-goal-observable" "door-close-v2-goal-observable" "basketball-v2-goal-observable" "shelf-place-v2-goal-observable" "button-press-v2-goal-observable" "button-press-topdown-v2-goal-observable" "faucet-close-v2-goal-observable" "faucet-open-v2-goal-observable" "handle-press-v2-goal-observable" "hammer-v2-goal-observable" "assembly-v2-goal-observable"
GPU_ID="0"
for task in "door-open-v2-goal-observable" "door-close-v2-goal-observable" "basketball-v2-goal-observable" "shelf-place-v2-goal-observable" 
# python org_results_mw.py --results_root "../results/results_AVDC_mw" 
# for task in "door-open-v2-goal-observable" 
do 
    CUDA_VISIBLE_DEVICES=$GPU_ID python benchmark_mw.py --env_name $task --n_exps 25 --ckpt_dir "../ckpts/metaworld" --milestone 24 --result_root "../results/results_AVDC_mw" 
done

# python org_results_mw.py --results_root "../results/results_AVDC_mw"
