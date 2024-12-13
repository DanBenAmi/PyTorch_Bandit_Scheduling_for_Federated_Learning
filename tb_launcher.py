import os
import subprocess


def launch_tensorboard(folders, starting_port=6008):
    processes = []
    for i, folder in enumerate(folders):
        port = starting_port + i
        logdir = os.path.abspath(folder)

        # Check if folder exists
        if not os.path.exists(logdir):
            print(f"Folder does not exist: {logdir}")
            continue

        # Command to launch TensorBoard
        command = ['tensorboard', '--logdir', logdir, '--port', str(port)]
        print(f"Launching TensorBoard for {logdir} on port {port}...")

        # Launch TensorBoard process
        process = subprocess.Popen(command)
        processes.append(process)

    return processes


if __name__ == "__main__":
    # List of folders containing TensorBoard logs
    folders = [
        'C:/Users/danbenam/PyTorch_Bandit_Scheduling_for_Federated_Learning/results/methods_compare/fashion_mnist/2024-09-10_15-14_non_iid__fashion_mnist__500_25__20t__lr5',
        'C:/Users/danbenam/PyTorch_Bandit_Scheduling_for_Federated_Learning/results/methods_compare/fashion_mnist/2024-09-10_15-13_non_iid__fashion_mnist__500_25__10t__lr5',
        'C:/Users/danbenam/PyTorch_Bandit_Scheduling_for_Federated_Learning/results/methods_compare/fashion_mnist/2024-09-10_15-10_non_iid__fashion_mnist__500_25__10t__lr5'
    ]

    # Launch TensorBoard instances
    launch_tensorboard(folders)
    tmp = 2