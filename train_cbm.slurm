#!/bin/bash
#SBATCH --job-name=skincbm_train
#SBATCH --output=skincbm_train_%j.out
#SBATCH --error=skincbm_train_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

echo "=================================================="
echo "SKINCBM TRAINING - Derm7pt Dataset"
echo "Training interpretable CBM for melanoma diagnosis"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Load necessary modules
module use /opt/nvidia/hpc_sdk/modulefiles
module load nvhpc-hpcx-cuda12/24.5

# Init conda
source /etc/profile.d/conda.sh

# Conda environment name
conda_env="CBM-env"

# Activate the Conda environment
conda activate $conda_env

# Set up paths
export PYTHONPATH="/home/csc29/projects/SynergyCBM/SkinCBM:$PYTHONPATH"
cd /home/csc29/projects/SynergyCBM/SkinCBM

# Print system information
echo "Job ID: $SLURM_JOB_ID"
echo "Process ID: $SLURM_PROCID running on $(hostname)"
echo "GPU Info:"
nvidia-smi
echo ""
echo "Python Environment:"
python3 --version
pip list | grep -E "(torch|torchvision|timm|numpy|pandas)"
echo ""

# Data paths
DATA_PATH="/home/xrai/datasets/derm7pt/release_v0"
OUTPUT_DIR="/home/csc29/projects/SynergyCBM/SkinCBM/trained_models/derm7pt_best"

# Verify paths exist
echo "Verifying data paths..."
if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: Data directory not found: $DATA_PATH"
    exit 1
fi

if [ ! -d "$DATA_PATH/images" ]; then
    echo "ERROR: Images directory not found: $DATA_PATH/images"
    exit 1
fi

if [ ! -d "$DATA_PATH/meta" ]; then
    echo "ERROR: Meta directory not found: $DATA_PATH/meta"
    exit 1
fi

echo "✓ Data directory: $DATA_PATH"
echo "✓ Images found: $(ls $DATA_PATH/images/*.jpg 2>/dev/null | wc -l) files"
echo "✓ Output dir: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting CBM training..."
echo "Configuration:"
echo "  Dataset: Derm7pt (7-point checklist)"
echo "  Backbone: ResNet50 (pretrained on ImageNet)"
echo "  Task Predictor: Linear (interpretable)"
echo "  Training Strategy: Joint (concepts + task together)"
echo "  Epochs: 50"
echo "  Batch Size: 16"
echo "  Learning Rate: 1e-4"
echo ""

# Run training
python3 examples/train_basic_cbm.py \
    --dataset derm7pt \
    --data_path "$DATA_PATH" \
    --backbone resnet50 \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --concept_loss_weight 1.0 \
    --output_dir "$OUTPUT_DIR" \
    --random_seed 42 \
    --run_information_analysis

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "TRAINING COMPLETED"
echo "=================================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: Model training completed"
    echo ""
    echo "Trained model saved to: $OUTPUT_DIR"
    echo "  - best_model.pth (best validation performance)"
    echo "  - training_history.json (loss curves)"
    echo "  - results.txt (final metrics)"
    echo ""
    echo "Next steps:"
    echo "  1. Use model in notebooks:"
    echo "     model = ConceptBottleneckModel.load('$OUTPUT_DIR/best_model.pth')"
    echo ""
    echo "  2. Run sample demo with trained model:"
    echo "     python3 examples/demo_sample_data.py --model_path $OUTPUT_DIR/best_model.pth"
    echo ""
    echo "  3. Evaluate on test set:"
    echo "     python3 examples/evaluate_model.py --model_path $OUTPUT_DIR/best_model.pth"
    echo ""
    echo "Model Performance Summary:"
    if [ -f "$OUTPUT_DIR/results.txt" ]; then
        cat "$OUTPUT_DIR/results.txt"
    fi
else
    echo "❌ FAILED: Training encountered errors"
    echo "Check error logs for details: skincbm_train_${SLURM_JOB_ID}.err"
fi

# Deactivate the Conda environment
conda deactivate

echo "=================================================="
