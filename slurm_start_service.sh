#!/bin/bash
#SBATCH --job-name=search-service
#SBATCH --gres=gpu:8
#SBATCH --mem=1024G
#SBATCH --cpus-per-task=128
#SBATCH --time=12000:00:00
#SBATCH --output=logs/service_%j.out
#SBATCH --error=logs/service_%j.err
# --partition=lowprio
# --qos=lowprio
# --reservation=moe

# Create directories if they don't exist
mkdir -p logs
mkdir -p runtime

# Server info file
SERVER_INFO_FILE="runtime/server_info_${SLURM_JOB_ID}.json"

# Cleanup function
cleanup() {
    echo "=========================================="
    echo "Service stopped at: $(date)"
    echo "=========================================="

    # Remove server info file
    if [ -f "$SERVER_INFO_FILE" ]; then
        rm -f "$SERVER_INFO_FILE"
        echo "Removed server info file: $SERVER_INFO_FILE"
    fi
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Starting at: $(date)"
echo "=========================================="

# Get hostname
HOSTNAME=$(hostname)
PORT=8000

# Write initial server info (status: loading)
cat > "$SERVER_INFO_FILE" << EOF
{
  "host": "$HOSTNAME",
  "port": $PORT,
  "url": "http://$HOSTNAME:$PORT",
  "job_id": "$SLURM_JOB_ID",
  "status": "loading",
  "started_at": "$(date -Iseconds)",
  "ready_at": null
}
EOF

echo "Server info file created: $SERVER_INFO_FILE"
echo "Status: loading"

# Set environment variables for multi-worker mode
export USE_GPU=true

# Start the FastAPI service with GPU support in background
/mnt/weka/home/shaurya.rohatgi/projects/faster_index/.conda/bin/python -m api.server \
    --use_gpu \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 16 &

SERVER_PID=$!

# Wait for server to be ready by polling the health endpoint
echo "Waiting for server to be ready..."
MAX_WAIT=300  # 5 minutes
WAIT_TIME=0
SLEEP_INTERVAL=5

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if curl -s -f "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Server is ready!"

        # Update server info (status: ready)
        cat > "$SERVER_INFO_FILE" << EOF
{
  "host": "$HOSTNAME",
  "port": $PORT,
  "url": "http://$HOSTNAME:$PORT",
  "job_id": "$SLURM_JOB_ID",
  "status": "ready",
  "started_at": "$(date -Iseconds)",
  "ready_at": "$(date -Iseconds)"
}
EOF

        echo "Server info updated: status=ready"
        break
    fi

    # Check if process is still running
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died during startup"
        exit 1
    fi

    sleep $SLEEP_INTERVAL
    WAIT_TIME=$((WAIT_TIME + SLEEP_INTERVAL))
    echo "Still waiting... ($WAIT_TIME/$MAX_WAIT seconds)"
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo "ERROR: Server did not become ready within $MAX_WAIT seconds"
    kill $SERVER_PID
    exit 1
fi

echo "=========================================="
echo "Server running at: http://$HOSTNAME:$PORT"
echo "Server info file: $SERVER_INFO_FILE"
echo "=========================================="

# Wait for the server process to finish
wait $SERVER_PID
