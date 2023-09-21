import subprocess
import time
import os
from collections import deque

def run_task_on_gpu(task, gpu_id):
    env = dict(CUDA_VISIBLE_DEVICES=str(gpu_id), **dict(os.environ))
    return subprocess.Popen(task, env=env, shell=True)

def main(tasks):
    gpu_count = 4  # You have 4 GPUs
    running_processes = [None] * gpu_count  # Initially, all GPUs are not running any processes

    tasks_queue = deque(tasks)

    while tasks_queue or any(p is not None for p in running_processes):
        for gpu_id in range(gpu_count):
            # Check if the process on this GPU has finished and if there's a new task to assign
            if (running_processes[gpu_id] is None or running_processes[gpu_id].poll() is not None) and tasks_queue:
                next_task = tasks_queue.popleft()
                print(f"Running task '{next_task}' on GPU {gpu_id}")
                running_processes[gpu_id] = run_task_on_gpu(next_task, gpu_id)
        
        time.sleep(1)  # Check every second. Adjust this value as needed.

if __name__ == "__main__":
    subsets = ['medical_flashcards', "wikidoc", "wikidoc_patient_information"]
    strategies = ["command", "instruct", "contrast", "instruct_rev", "contrast_rev", "original", "mask", "remove"]
    scales = ["7B", "13B"]

    # Generating the task list
    tasks = [f"python eval.py --subset {subset} --strategy {strategy} --scale {scale}" for strategy in strategies for scale in scales for subset in subsets]
    print(tasks)

    main(tasks)
