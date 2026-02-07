import subprocess
import os
from pathlib import Path
import numpy as np

testing = os.name != 'posix'

limit = 10 if testing else 10 - int(
    subprocess.run(
        "qstat | grep iferreira | wc -l",
        shell=True,
        capture_output=True,
        text=True
    ).stdout.strip()
)
total = 0

def generate_pbs_script(python_cmd, experiment_name):
    if testing: return

    template = Path('run.job').read_text()
    pbs_script = template.format(
        experiment_name=experiment_name,
        python_cmd=python_cmd
    )
    temp_file = Path("temp_pbs_script.job")
    temp_file.write_text(pbs_script)

    try:
        result = subprocess.run(['qsub', str(temp_file)], capture_output=True, text=True)
        print(f"Job submitted: {result.stdout.strip()}")
        if result.stderr:
            print(f"Errors: {result.stderr.strip()}")
    finally:
        temp_file.unlink(missing_ok=True)


def is_training_complete(dataset, method, model_name, seed, teacher_model=None):
    """Check if training is already completed for a given configuration."""
    import json
    if method == 'pure':
        status_path = Path(f'models/{dataset}/pure/{model_name}/{seed}/status.json')
    else:
        status_path = Path(f'models/{dataset}/{method}/{teacher_model}_to_{model_name}/{seed}/status.json')
    
    if status_path.exists():
        with open(status_path, 'r') as f:
            status = json.load(f)
            return status.get('status') == 'completed'
    return False


def check_path_and_skip(model, dataset, seed, distillation='none', teacher_model=None):
    """
    Check if training is already completed for this configuration.
    Returns True if we should skip (already completed), False if we should run.
    """
    global total, limit
    if total == limit: 
        print('Queue limit reached, exiting')
        exit()

    # Check the status.json in the model directory
    if is_training_complete(dataset, 'pure' if distillation == 'none' else distillation, 
                            model, seed, teacher_model):
        return True

    total += 1
    return False

def generate_python_cmd(experiment_name, model, dataset, distillation='none', 
                        teacher_model=None, teacher_weights=None, alpha=0.5, temperature=4.0):
    """Generate the python command for training."""
    cmd = f"python train.py --experiment_name {experiment_name} --model {model} --dataset {dataset}"
    
    if distillation != 'none':
        cmd += f" --distillation {distillation}"
        cmd += f" --teacher_model {teacher_model}"
        cmd += f" --teacher_weights {teacher_weights}"
        cmd += f" --alpha {alpha}"
        if distillation == 'logit':
            cmd += f" --temperature {temperature}"
    
    print(cmd)
    return cmd


def get_teacher_weights_path(dataset, teacher_model, seed):
    """Get the path to a trained teacher model (best.pth)."""
    return f"models/{dataset}/pure/{teacher_model}/{seed}/best.pth"


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

runs = 3
dataset = 'Cifar10'

# --- Pure training (no distillation) ---
pure_models = ['ResNet56']
for run in range(runs):
    for model in pure_models:
        if check_path_and_skip(model, dataset, run): continue
        experiment_name = f'{dataset}/pure/{model}/{run}'
        python_cmd = generate_python_cmd(experiment_name, model, dataset)
        generate_pbs_script(python_cmd, experiment_name)

# --- Distillation experiments ---
# Uncomment and configure as needed

# Logit distillation: ResNet112 -> ResNet20
teacher_model = 'ResNet112'
student_model = 'ResNet56'
for run in range(runs):
    if check_path_and_skip(student_model, dataset, run, distillation='logit', teacher_model=teacher_model): continue
    teacher_weights = get_teacher_weights_path(dataset, teacher_model, run)
    experiment_name = f'{dataset}/logit/{teacher_model}_to_{student_model}/{run}'
    python_cmd = generate_python_cmd(
        experiment_name, student_model, dataset,
        distillation='logit', teacher_model=teacher_model,
        teacher_weights=teacher_weights, alpha=0.5, temperature=4.0
    )
    generate_pbs_script(python_cmd, experiment_name)

# Factor transfer: ResNet112 -> ResNet20
for run in range(runs):
    if check_path_and_skip(student_model, dataset, run, distillation='factor_transfer', teacher_model=teacher_model): continue
    teacher_weights = get_teacher_weights_path(dataset, teacher_model, run)
    experiment_name = f'{dataset}/factor_transfer/{teacher_model}_to_{student_model}/{run}'
    python_cmd = generate_python_cmd(
        experiment_name, student_model, dataset,
        distillation='factor_transfer', teacher_model=teacher_model,
        teacher_weights=teacher_weights, alpha=0.5
    )
    generate_pbs_script(python_cmd, experiment_name)


print('All experiments are finished / queued')


# ============================================================================
# TEST COMMANDS - Run these manually to verify everything works
# ============================================================================
# 
# NEW DIRECTORY STRUCTURE:
#   models/
#   └── Cifar10/
#       ├── pure/
#       │   └── ResNet56/
#       │       └── 0/
#       │           ├── best.pth        # Best model weights (for inference/teacher)
#       │           ├── checkpoint.pth  # Full training state (for resume)
#       │           └── status.json     # Training status
#       └── logit/
#           └── ResNet56_to_ResNetBaby/
#               └── 0/
#                   ├── best.pth
#                   ├── checkpoint.pth
#                   └── status.json
#
# RESUMPTION BEHAVIOR:
#   - If status.json says "completed", training exits immediately
#   - If checkpoint.pth exists, training resumes from last epoch
#   - Use --force_restart to ignore checkpoints and train from scratch
#
# Quick test commands:
# 
# 1. PURE TRAINING - Train a teacher model (ResNet56):
#    python train.py --experiment_name test/pure/ResNet56/0 --model ResNet56 --dataset Cifar10
#
# 2. RESUME INTERRUPTED TRAINING (automatic if checkpoint exists):
#    python train.py --experiment_name test/pure/ResNet56/0 --model ResNet56 --dataset Cifar10
#    (Will automatically resume from where it left off)
#
# 3. FORCE RESTART (ignore existing checkpoint):
#    python train.py --experiment_name test/pure/ResNet56/0 --model ResNet56 --dataset Cifar10 --force_restart
#
# 4. LOGIT DISTILLATION - ResNet56 teacher -> ResNetBaby student:
#    python train.py --experiment_name test/logit/ResNet56_to_ResNetBaby/0 \
#        --model ResNetBaby --dataset Cifar10 \
#        --distillation logit --teacher_model ResNet56 \
#        --teacher_weights models/Cifar10/pure/ResNet56/0/best.pth \
#        --alpha 0.5 --temperature 4.0
#
# 5. FACTOR TRANSFER - ResNet56 teacher -> ResNetBaby student:
#    python train.py --experiment_name test/factor_transfer/ResNet56_to_ResNetBaby/0 \
#        --model ResNetBaby --dataset Cifar10 \
#        --distillation factor_transfer --teacher_model ResNet56 \
#        --teacher_weights models/Cifar10/pure/ResNet56/0/best.pth \
#        --alpha 0.5
#
# 6. DIFFERENT ALPHA VALUES:
#    python train.py --experiment_name test/logit_alpha0.3/ResNet56_to_ResNetBaby/0 \
#        --model ResNetBaby --dataset Cifar10 \
#        --distillation logit --teacher_model ResNet56 \
#        --teacher_weights models/Cifar10/pure/ResNet56/0/best.pth \
#        --alpha 0.3 --temperature 4.0
#
# ============================================================================