# Training on HPC

## Quick Start

Submit the training job:
```bash
cd /home/csc29/projects/SynergyCBM/SkinCBM
sbatch train_cbm.slurm
```

Monitor the job:
```bash
# Check job status
squeue -u $USER

# Watch output in real-time
tail -f skincbm_train_*.out

# Check for errors
tail -f skincbm_train_*.err
```

## What It Does

The SLURM script trains a Concept Bottleneck Model on the Derm7pt dataset with:
- **Dataset**: 2,000 dermoscopy images from `/home/xrai/datasets/derm7pt/release_v0`
- **Concepts**: 7-point checklist (pigment network, blue-whitish veil, etc.)
- **Architecture**: ResNet50 backbone + Linear task predictor
- **Training**: 50 epochs, ~10 minutes on V100 GPU
- **Output**: Best model saved to `trained_models/derm7pt_best/`

## Expected Performance

- **Concept Accuracy**: 75-80%
- **Task F1 Score**: 68-72%
- **Training Time**: ~10 minutes

## Using the Trained Model

Once training completes, use the model:

### In Python Scripts
```python
from src.models.basic_cbm import ConceptBottleneckModel

model = ConceptBottleneckModel.load('trained_models/derm7pt_best/best_model.pth')
```

### In Notebooks
```python
# Load trained model
model = ConceptBottleneckModel.load('../trained_models/derm7pt_best/best_model.pth')

# Run inference
concepts, logits = model(image)
```

### Demo with Trained Model
```bash
python3 examples/demo_sample_data.py \
    --model_path trained_models/derm7pt_best/best_model.pth
```

## Output Files

After training, you'll find:
- `trained_models/derm7pt_best/best_model.pth` - Best checkpoint
- `trained_models/derm7pt_best/training_history.json` - Loss curves
- `trained_models/derm7pt_best/results.txt` - Final metrics
- `skincbm_train_<job_id>.out` - Training log
- `skincbm_train_<job_id>.err` - Error log (if any)

## Troubleshooting

### Job Won't Start
```bash
# Check queue
squeue -u $USER

# Check job details
scontrol show job <job_id>
```

### Out of Memory
Reduce batch size in the script:
```bash
# Edit train_cbm.slurm, change:
--batch_size 16
# to:
--batch_size 8
```

### Dataset Not Found
Verify the dataset path:
```bash
ls /home/xrai/datasets/derm7pt/release_v0/images/*.jpg | wc -l
# Should show ~2000 images
```

### Module Errors
Check if conda environment exists:
```bash
conda env list | grep CBM-env
```

## Customization

Edit `train_cbm.slurm` to change:
- **Time limit**: `#SBATCH --time=4:00:00`
- **Memory**: `#SBATCH --mem=32G`
- **Epochs**: `--epochs 50`
- **Batch size**: `--batch_size 16`
- **Learning rate**: `--learning_rate 1e-4`
- **Output directory**: `OUTPUT_DIR=...`

## Alternative: Quick Test Run

For a quick test (10 epochs, ~2 minutes):
```bash
python3 examples/train_basic_cbm.py \
    --data_path /home/xrai/datasets/derm7pt/release_v0 \
    --epochs 10 \
    --batch_size 16 \
    --output_dir ./outputs/quick_test
```
