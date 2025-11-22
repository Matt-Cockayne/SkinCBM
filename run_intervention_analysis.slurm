#!/bin/bash
#SBATCH --job-name=intervention_analysis
#SBATCH --output=intervention_analysis_%j.out
#SBATCH --error=intervention_analysis_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

echo "=================================================="
echo "CONCEPT INTERVENTION ANALYSIS - Test Set"
echo "Analyzing intervention effects on Derm7pt test set"
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
echo "GPU Info:"
nvidia-smi
echo ""
echo "Python Environment:"
python3 --version
pip list | grep -E "(torch|torchvision|numpy|pandas|matplotlib)"
echo ""

# Paths
MODEL_PATH="trained_models/derm7pt_best/best_model.pth"
DATA_PATH="/home/xrai/datasets/derm7pt/release_v0"
OUTPUT_DIR="outputs/intervention_analysis"

# Verify paths exist
echo "Verifying paths..."
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found: $MODEL_PATH"
    exit 1
fi

if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: Data directory not found: $DATA_PATH"
    exit 1
fi

echo "✓ Model: $MODEL_PATH"
echo "✓ Data: $DATA_PATH"
echo "✓ Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting intervention analysis..."
echo "Configuration:"
echo "  Batch Size: 32"
echo "  Workers: 4"
echo "  Analysis: Individual + Cumulative interventions"
echo "  Output: CSVs + Visualizations"
echo ""

# Run intervention analysis
python3 examples/intervention_analysis.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 32 \
    --num_workers 4

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "ANALYSIS COMPLETED"
echo "=================================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: Intervention analysis completed"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    if [ -f "$OUTPUT_DIR/full_intervention_results.csv" ]; then
        ROWS=$(wc -l < "$OUTPUT_DIR/full_intervention_results.csv")
        echo "  ✓ full_intervention_results.csv ($((ROWS-1)) samples)"
    fi
    if [ -f "$OUTPUT_DIR/overall_summary.csv" ]; then
        echo "  ✓ overall_summary.csv"
        echo ""
        echo "Overall Summary:"
        cat "$OUTPUT_DIR/overall_summary.csv"
        echo ""
    fi
    if [ -f "$OUTPUT_DIR/per_concept_impact.csv" ]; then
        echo "  ✓ per_concept_impact.csv"
        echo ""
        echo "Top 3 Most Impactful Concepts:"
        head -n 4 "$OUTPUT_DIR/per_concept_impact.csv" | tail -n 3
        echo ""
    fi
    if [ -f "$OUTPUT_DIR/misclassified_summary.csv" ]; then
        echo "  ✓ misclassified_summary.csv"
    fi
    if [ -f "$OUTPUT_DIR/intervention_effects_distribution.png" ]; then
        SIZE=$(du -h "$OUTPUT_DIR/intervention_effects_distribution.png" | cut -f1)
        echo "  ✓ intervention_effects_distribution.png ($SIZE)"
    fi
    if [ -f "$OUTPUT_DIR/cumulative_intervention_analysis.png" ]; then
        SIZE=$(du -h "$OUTPUT_DIR/cumulative_intervention_analysis.png" | cut -f1)
        echo "  ✓ cumulative_intervention_analysis.png ($SIZE)"
    fi
    echo ""
    echo "Next steps:"
    echo "  - Review per_concept_impact.csv to identify most important concepts"
    echo "  - Check intervention_effects_distribution.png for concept importance"
    echo "  - Use cumulative_intervention_analysis.png for fix rate analysis"
    echo "  - Import full_intervention_results.csv into papers for detailed tables"
else
    echo "❌ FAILED: Analysis encountered errors"
    echo "Check error logs: intervention_analysis_${SLURM_JOB_ID}.err"
fi

# Deactivate the Conda environment
conda deactivate

echo "=================================================="
