import subprocess
import os
import json
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


# ============================================================================
# EXPERIMENT TRACKER
# ============================================================================

class ExperimentTracker:
    def __init__(self, queue_limit):
        self.queue_limit = queue_limit
        self.completed = []
        self.running = []       # list of (name, job_info)
        self.queued = []        # submitted this run
        self.pending = []       # not yet queued (hit limit or waiting on teacher)

    @property
    def total(self):
        return (len(self.completed) + len(self.running) + len(self.queued)
                + len(self.pending))

    def record(self, name, status, job_info=None):
        if status == 'running':
            self.running.append((name, job_info))
        else:
            {'completed': self.completed,
             'queued': self.queued, 'pending': self.pending}[status].append(name)

    def summary(self):
        t = self.total
        c = len(self.completed)
        pct = (c / t * 100) if t else 0
        print(f'EXPERIMENT CHARLIE — {c}/{t} completed ({pct:.1f}%)')
        print(f'  Finished:           {c}')
        print(f'  Not started:        {len(self.pending)}')
        print(f'  Already on PBS:     {len(self.running)}')
        if self.running:
            for name, info in self.running:
                state = info.get('state', '?') if info else '?'
                elapsed = info.get('elapsed', '') if info else ''
                tag = f'[{state}]' if not elapsed else f'[{state} {elapsed}]'
                print(f'    - {tag} {name}')
        print(f'  Queued this run:    {len(self.queued)}')
        if self.queued:
            for name in self.queued:
                print(f'    - {name}')

tracker = ExperimentTracker(limit)


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def generate_pbs_script(python_cmd, experiment_name):
    if testing: return

    template = (Path(__file__).parent / 'run.job').read_text()
    pbs_script = template.format(
        experiment_name=experiment_name,
        python_cmd=python_cmd
    )
    temp_file = Path("temp_pbs_script.job")
    temp_file.write_text(pbs_script)

    try:
        result = subprocess.run(['qsub', str(temp_file)], capture_output=True, text=True)
        job_id = result.stdout.strip()
        print(f"Job submitted: {job_id}")
        if result.stderr:
            print(f"Errors: {result.stderr.strip()}")

        # Store PBS job ID in status.json for orphan detection
        if job_id:
            status_path = Path(f'experiments/{experiment_name}/status.json')
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status = {}
            if status_path.exists():
                with open(status_path) as f:
                    status = json.load(f)
            status['pbs_job_id'] = job_id
            with open(status_path, 'w') as f:
                json.dump(status, f, indent=2)
    finally:
        temp_file.unlink(missing_ok=True)


def _get_job_info(job_id):
    """Get PBS job info. Returns dict with 'alive', 'state', 'elapsed' or None."""
    try:
        result = subprocess.run(['qstat', '-f', job_id], capture_output=True, text=True)
        if result.returncode != 0:
            return None
        info = {'alive': True, 'state': '?', 'elapsed': ''}
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith('job_state ='):
                info['state'] = line.split('=')[1].strip()
            elif line.startswith('stime ='):
                info['elapsed'] = _elapsed_since(line.split('=', 1)[1].strip())
            elif line.startswith('resources_used.walltime ='):
                info['elapsed'] = line.split('=')[1].strip()
        return info
    except FileNotFoundError:
        return None


def _elapsed_since(time_str):
    """Parse PBS time string and return elapsed duration."""
    from datetime import datetime
    try:
        start = datetime.strptime(time_str, '%a %b %d %H:%M:%S %Y')
        delta = datetime.now() - start
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f'{hours}:{minutes:02d}:{seconds:02d}'
    except (ValueError, TypeError):
        return time_str


def get_experiment_status(dataset, method, model_name, seed, teacher_model=None, alpha=None):
    """Returns ('completed', None), ('running', job_info), or ('pending', None)."""
    if method == 'pure':
        status_path = Path(f'experiments/{dataset}/pure/{model_name}/{seed}/status.json')
    else:
        status_path = Path(f'experiments/{dataset}/{method}/alpha_{alpha}/{teacher_model}_to_{model_name}/{seed}/status.json')

    if not status_path.exists():
        return 'pending', None

    with open(status_path) as f:
        status = json.load(f)

    if status.get('status') == 'completed':
        return 'completed', None

    # Check if there's a live PBS job (queued or running on the cluster)
    job_id = status.get('pbs_job_id')
    if job_id:
        job_info = _get_job_info(job_id)
        if job_info:
            return 'running', job_info

    return 'pending', None


def is_training_complete(dataset, method, model_name, seed, teacher_model=None, alpha=None):
    """Check if training is already completed for a given configuration."""
    status, _ = get_experiment_status(dataset, method, model_name, seed, teacher_model, alpha)
    return status == 'completed'


def check_path_and_skip(model, dataset, seed, distillation='none', teacher_model=None, alpha=None):
    """Check experiment status, record it, and return True to skip or False to queue."""
    method = 'pure' if distillation == 'none' else distillation
    name = get_experiment_name(dataset, model, seed, distillation, teacher_model, alpha)
    status, job_info = get_experiment_status(dataset, method, model, seed, teacher_model, alpha)

    if status == 'completed':
        tracker.record(name, 'completed')
        return True
    if status == 'running':
        tracker.record(name, 'running', job_info)
        return True

    # Pending — queue it if we have capacity, otherwise mark as pending
    if len(tracker.queued) >= tracker.queue_limit:
        tracker.record(name, 'pending')
        return True  # skip queueing but still counted
    tracker.record(name, 'queued')
    return False


def generate_python_cmd(model, dataset, seed, distillation='none',
                        teacher_model=None, teacher_weights=None, alpha=0.5, temperature=4.0):
    """Generate the python command for training."""
    cmd = f"python toolbox/train.py --model {model} --dataset {dataset} --seed {seed}"

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


def get_experiment_name(dataset, model, seed, distillation='none', teacher_model=None, alpha=None):
    """Get the experiment name/path for PBS job naming."""
    if distillation == 'none':
        return f'{dataset}/pure/{model}/{seed}'
    else:
        return f'{dataset}/{distillation}/alpha_{alpha}/{teacher_model}_to_{model}/{seed}'


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
# Pure baselines queue before distillation so teacher weights exist.
# The check_path_and_skip mechanism ensures completed runs are skipped.
# ============================================================================

runs = 3
alphas = [0.25, 0.5, 0.75, 0.95]
datasets = ['Cifar100', 'Cifar10', 'SVHN', 'TinyImageNet']
teacher_model = 'ResNet112'
student_models = ['ResNet56']
# distillation_methods = ['logit', 'factor_transfer', 'attention_transfer', 'fitnets', 'rkd', 'nst']
distillation_methods = ['logit', 'factor_transfer', 'attention_transfer', 'fitnets', 'nst'] # Disabled RKD for now, it's broken

for run in range(runs):
    for dataset in datasets:
        # --- Pure training (teacher + student baselines) ---
        for model in [teacher_model] + student_models:
            if check_path_and_skip(model, dataset, run): continue
            experiment_name = get_experiment_name(dataset, model, run)
            python_cmd = generate_python_cmd(model, dataset, run)
            generate_pbs_script(python_cmd, experiment_name)

        # --- Distillation: teacher -> each student, all methods, all alphas ---
        for student_model in student_models:
            for method in distillation_methods:
                for alpha in alphas:
                    # Verify teacher is fully trained before queuing any KD student
                    name = get_experiment_name(dataset, student_model, run,
                                               distillation=method, teacher_model=teacher_model,
                                               alpha=alpha)
                    if not is_training_complete(dataset, 'pure', teacher_model, run):
                        tracker.record(name, 'pending')
                        continue
                    if check_path_and_skip(student_model, dataset, run,
                                           distillation=method, teacher_model=teacher_model,
                                           alpha=alpha): continue
                    teacher_weights = get_teacher_weights_path(dataset, teacher_model, run)
                    experiment_name = name
                    python_cmd = generate_python_cmd(
                        student_model, dataset, run,
                        distillation=method, teacher_model=teacher_model,
                        teacher_weights=teacher_weights, alpha=alpha, temperature=4.0
                    )
                    generate_pbs_script(python_cmd, experiment_name)

tracker.summary()
