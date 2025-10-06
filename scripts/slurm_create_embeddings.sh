#!/bin/bash
#SBATCH --job-name=create_embeddings
#SBATCH --array=0-7                      # 8 jobs (0-7)
#SBATCH --gres=gpu:8                     # 8 GPUs per job
#SBATCH --mem-per-cpu=15G
#SBATCH --cpus-per-task=32               # More CPUs for data loading
#SBATCH --time=12000:00:00
#SBATCH --output=logs/embeddings_%A_%a.out
#SBATCH --error=logs/embeddings_%A_%a.err
#SBATCH --requeue                        # Automatically requeue on preemption
#SBATCH --signal=TERM@300                # Send SIGTERM 300 seconds before time limit
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --distribution=pack
#SBATCH --reservation=moe

# Print job info
echo "=========================================="
echo "SLURM Job Info"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run the embedding creation script
# Each job processes its assigned chunk of files
/mnt/weka/home/shaurya.rohatgi/projects/faster_index/.conda/bin/python 03_create_embeddings.py \
    --passages_dir /mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_data/outputs/passages \
    --output_dir /mnt/weka/shrd/k2m/shaurya.rohatgi/faster_index_data/outputs \
    --model_name facebook/contriever \
    --batch_size 65536 \
    --max_length 128 \
    --num_workers 32 \
    --total_jobs 8 \
    --job_id $SLURM_ARRAY_TASK_ID \

# Print completion info
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
