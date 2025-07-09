###############################################################################
# Run every model × PLE × dataset experiment serially.
#
# • Each run uses *all four* GPUs via torchrun (`nproc_per_node=4`).
# • Output for every run goes to      logs/<model>_PLE<flag>_<data>.log
#   – the file is TRUNCATED once at the start of the run, then all lines
#     (header, torchrun stdout/stderr, trailer) are appended.
# • If torchrun crashes or times-out the script records the exit-code
#   and continues to the next configuration.
###############################################################################

set -u                                    # error on undefined variables
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

# -------- parameter grids ---------------------------------------------------
models=( "GIN" "GCN-RNI" "GCN-Pooling" "GMN" )
ple_flags=( "False" "True" )
datasets=( "syn" "sr" "cfi" "3xor" "exp" )

# -------- optional per-run wall-clock limit ---------------------------------
TIME_LIMIT_H=2            # hours; adjust or set to 0 to disable

# -------- loop --------------------------------------------------------------
job_id=0
for model in "${models[@]}"; do
  for ple in "${ple_flags[@]}"; do
    for data in "${datasets[@]}"; do

      run_name="${model}_PLE${ple}_${data}"
      log_file="${LOG_DIR}/${run_name}.log"

      # truncate / create a fresh log file
      : > "${log_file}"

      echo "===== [$(date '+%F %T')]  START ${run_name}  =====" \
           | tee -a "${log_file}"

      # build the torchrun command
      cmd=( torchrun --nproc_per_node=4
            Algorithmic_Alignment/run.py
            --model "${model}" --PLE "${ple}" --data "${data}" )

      # run with optional timeout
      if (( TIME_LIMIT_H > 0 )); then
        timeout --signal=SIGINT "$(( TIME_LIMIT_H * 3600 ))"  \
                "${cmd[@]}"    >> "${log_file}" 2>&1
        rc=$?
      else
        "${cmd[@]}"            >> "${log_file}" 2>&1
        rc=$?
      fi

      echo "[end $(date '+%F %T')] exit_code=${rc}" >> "${log_file}"
      echo "Finished ${run_name}  (rc=${rc})"

      # brief pause so NVIDIA driver fully releases memory
      sleep 10
      (( job_id++ ))
    done
  done
done

echo -e "\n*** All experiments finished at $(date '+%F %T') ***"

