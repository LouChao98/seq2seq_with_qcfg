function envsource
  for line in (cat $argv | grep -v '^#' | sed '/^[[:space:]]*$/d')
    set item (string split -m 1 '=' $line)
    set -gx $item[1] $item[2]
    echo "Exported key $item[1]"
  end
end

function sbatch24
    envsource ./.env
    echo project=$PROJECT
    sbatch --job-name $PROJECT \
        --partition critical \
        --nodes 1 \
        --time 1-0:0:0 \
        --cpus-per-task 6 \
        --mail-type all \
        --mail-user louchao@shanghaitech.edu.cn \
        --gres gpu:NVIDIATITANRTX:1 \
        --output logs/slurm/%j.out \
        --error logs/slurm/%j.err \
        --exclude=ai_gpu34 \
        --wrap "source $HOME/.local/share/slurm_python_env.sh && python3 $argv"
end

# TODO I may choose different types of 12gb gpus by the techique in the following url
# For now, I just use NVIDIATITANV
# https://stackoverflow.com/questions/66945680/selecting-from-multiple-slurm-gpu-resources
function sbatch12
    envsource ./.env
    echo project=$PROJECT
    sbatch --job-name $PROJECT \
        --partition critical \
        --nodes 1 \
        --time 1-0:0:0 \
        --cpus-per-task 6 \
        --mail-type all \
        --mail-user louchao@shanghaitech.edu.cn \
        --gres gpu:NVIDIATITANV:1 \
        --output logs/slurm/%j.out \
        --error logs/slurm/%j.err \
        --exclude=ai_gpu34 \
        --wrap "source $HOME/.local/share/slurm_python_env.sh && python3 $argv"
end


function sbatchany
    envsource ./.env
    echo project=$PROJECT
    sbatch --job-name $PROJECT \
        --partition critical \
        --nodes 1 \
        --time 1-0:0:0 \
        --cpus-per-task 6 \
        --mail-type all \
        --mail-user louchao@shanghaitech.edu.cn \
        --gres gpu:1 \
        --output logs/slurm/%j.out \
        --error logs/slurm/%j.err \
        --exclude=ai_gpu34 \
        --wrap "source $HOME/.local/share/slurm_python_env.sh && python3 $argv"
end
