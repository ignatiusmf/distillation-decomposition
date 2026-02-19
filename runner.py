import subprocess
import os
from pathlib import Path
import numpy as np

testing = os.name != 'posix' or not os.path.isfile('/usr/bin/qsub')

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
        status_path = Path(f'experiments/{dataset}/pure/{model_name}/{seed}/status.json')
    else:
        status_path = Path(f'experiments/{dataset}/{method}/{teacher_model}_to_{model_name}/{seed}/status.json')

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

    # Check the status.json in the experiment directory
    if is_training_complete(dataset, 'pure' if distillation == 'none' else distillation,
                            model, seed, teacher_model):
        return True

    total += 1
    return False

def generate_python_cmd(model, dataset, seed, distillation='none',
                        teacher_model=None, teacher_weights=None, alpha=0.5, temperature=4.0):
    """Generate the python command for training."""
    cmd = f"python train.py --model {model} --dataset {dataset} --seed {seed}"

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
    return f"experiments/{dataset}/pure/{teacher_model}/{seed}/best.pth"


def get_experiment_name(dataset, model, seed, distillation='none', teacher_model=None):
    """Get the experiment name/path for PBS job naming."""
    if distillation == 'none':
        return f'{dataset}/pure/{model}/{seed}'
    else:
        return f'{dataset}/{distillation}/{teacher_model}_to_{model}/{seed}'


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
# Pure baselines queue before distillation so teacher weights exist.
# The check_path_and_skip mechanism ensures completed runs are skipped.
# ============================================================================

runs = 1
datasets = ['Cifar100', 'Cifar10', 'SVHN', 'TinyImageNet']
teacher_model = 'ResNet112'
student_models = ['ResNet56']
distillation_methods = ['logit', 'factor_transfer', 'attention_transfer', 'fitnets', 'rkd', 'nst']

for dataset in datasets:
    # --- Pure training (teacher + student baselines) ---
    for model in [teacher_model] + student_models:
        for run in range(runs):
            if check_path_and_skip(model, dataset, run): continue
            experiment_name = get_experiment_name(dataset, model, run)
            python_cmd = generate_python_cmd(model, dataset, run)
            generate_pbs_script(python_cmd, experiment_name)

    # --- Distillation: teacher -> each student, all methods ---
    for student_model in student_models:
        for method in distillation_methods:
            for run in range(runs):
                if check_path_and_skip(student_model, dataset, run,
                                       distillation=method, teacher_model=teacher_model): continue
                teacher_weights = get_teacher_weights_path(dataset, teacher_model, run)
                experiment_name = get_experiment_name(dataset, student_model, run,
                                                     distillation=method, teacher_model=teacher_model)
                python_cmd = generate_python_cmd(
                    student_model, dataset, run,
                    distillation=method, teacher_model=teacher_model,
                    teacher_weights=teacher_weights, alpha=0.5, temperature=4.0
                )
                generate_pbs_script(python_cmd, experiment_name)

print('All experiments are finished / queued')
