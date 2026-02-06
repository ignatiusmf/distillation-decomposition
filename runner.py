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

def check_path_and_skip(experiment_name):
    experiment_path = Path(f'experiments/{experiment_name}')
    global total, limit
    if total == limit: 
        print('Queue limit reached, exiting')
        exit()

    if experiment_path.exists():
        return True

    experiment_path.mkdir(parents=True)
    total += 1
    return False

def generate_python_cmd(experiment_name, model, dataset):
    output = f"python train.py --experiment_name {experiment_name} --model {model} --dataset {dataset}"
    print(output)
    return output

runs = 3
models = ['ResNet56']
dataset = 'Cifar10'
for run in range(runs):
    for model in models:
        experiment_name = f'{dataset}/{model}/{run}'
        if check_path_and_skip(experiment_name): continue
        python_cmd = generate_python_cmd(experiment_name, model, dataset)
        generate_pbs_script(python_cmd, experiment_name)


print('All experiments are finished / queued')