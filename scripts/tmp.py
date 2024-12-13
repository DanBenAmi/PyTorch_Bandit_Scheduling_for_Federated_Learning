C:\Users\danbenam\.conda\envs\torch_env\python.exe - X
pycache_prefix = C:\Users\danbenam\AppData\Local\JetBrains\PyCharmCE2024
.2\cpython - cache
"C:/Users/danbenam/AppData/Local/JetBrains/PyCharm Community Edition 2024.2/plugins/python-ce/helpers/pydev/pydevd.py" - -multiprocess - -qt - support = auto - -client
127.0
.0
.1 - -port
2970 - -file
C:\Users\danbenam\PyTorch_Bandit_Scheduling_for_Federated_Learning\scripts\res_plot_loss_regret.py
Connected
to
pydev
debugger(build
242.20224
.347)
..\results\methods_compare\selected_res\2024 - 11 - 23_15 - 27
_non_iid__lin_reg__25_5__25t__lr2
Backend
TkAgg is interactive
backend.Turning
interactive
mode
on.
plot_data(data_dicts)
plt.show(block=True)

PyDev
console: starting.

np.max(np.array([data_dicts[cs]["loss"] for cs in data_dicts.keys()]))
Traceback(most
recent
call
last):
File
"C:\Users\danbenam\AppData\Local\JetBrains\PyCharm Community Edition 2024.2\plugins\python-ce\helpers\pydev\_pydevd_bundle\pydevd_exec2.py", line
3, in Exec
exec(exp, global_vars, local_vars)
File
"<input>", line
1, in < module >
ValueError: setting
an
array
element
with a sequence.The requested array has an inhomogeneous shape after 1 dimensions.The detected shape was (5, ) + inhomogeneous part.
np.max(np.stack([data_dicts[cs]["loss"] for cs in data_dicts.keys()]))
Traceback(most
recent
call
last):
File
"C:\Users\danbenam\AppData\Local\JetBrains\PyCharm Community Edition 2024.2\plugins\python-ce\helpers\pydev\_pydevd_bundle\pydevd_exec2.py", line
3, in Exec
exec(exp, global_vars, local_vars)
File
"<input>", line
1, in < module >
File
"<__array_function__ internals>", line
200, in stack
File
"C:\Users\danbenam\.conda\envs\torch_env\lib\site-packages\numpy\core\shape_base.py", line
464, in stack
raise ValueError('all input arrays must have the same shape')
ValueError: all
input
arrays
must
have
the
same
shape
np.max(np.concatenate([data_dicts[cs]["loss"] for cs in data_dicts.keys()]))
1.4646589756011963
np.min(np.concatenate([data_dicts[cs]["loss"] for cs in data_dicts.keys()]))
0.02677491120994091
max_l = np.max(np.concatenate([data_dicts[cs]["loss"] for cs in data_dicts.keys()]))
min_l = np.min(np.concatenate([data_dicts[cs]["loss"] for cs in data_dicts.keys()]))
for cs in data_dicts.keys():
    data_dicts[cs]["loss"] = data_dicts[cs]["loss"] / max_l

data_dicts = read_pkl_files(dir_path)
plot_data(data_dicts)

data_dicts = read_pkl_files(dir_path)
max_l = np.max(np.concatenate([data_dicts[cs]["loss"] for cs in data_dicts.keys()]))
min_l = np.min(np.concatenate([data_dicts[cs]["loss"] for cs in data_dicts.keys()]))
for cs in data_dicts.keys():
    data_dicts[cs]["loss"] = np.array(data_dicts[cs]["loss"]) / max_l

plot_data(data_dicts)
data_dicts = read_pkl_files(dir_path)
max_l = np.max(np.concatenate([data_dicts[cs]["loss"] for cs in data_dicts.keys()]))
min_l = np.min(np.concatenate([data_dicts[cs]["loss"] for cs in data_dicts.keys()]))
for cs in data_dicts.keys():
    data_dicts[cs]["loss"] = np.array(data_dicts[cs]["loss"]) / (max_l + 0.2)

plot_data(data_dicts)


def plot_data(data_dicts, window_size=1, save_path=None):
    plt.figure(figsize=(16, 6))

    # Plot time vs. accuracy
    plt.subplot(1, 2, 1)
    markers = ['o', '+', '*', 'x', 'v']
    for i, (cs_name, data) in enumerate(data_dicts.items()):
        smoothed_reg = moving_average(np.cumsum(data['regret'][1:]), window_size)
        plt.plot(np.linspace(0, data["time"][-1], len(smoothed_reg)), smoothed_reg, marker=markers[i],
                 label=f"{cs_name}", markevery=len(data_dicts[cs_name]["regret"]) // len(data_dicts[cs_name]["time"])
        plt.xlabel('Time', fontsize=FONTSIZE)
        plt.ylabel('Regret', fontsize=FONTSIZE)
        # plt.title('Time vs. Regret', fontsize=FONTSIZE)
        plt.legend(fontsize=FONTSIZE)

        # Plot time vs. loss
        plt.subplot(1, 2, 2)
        for i, (cs_name, data) in enumerate(data_dicts.items()):
            smoothed_loss = moving_average(data['loss'], window_size)
        plt.plot(data['time'][:len(smoothed_loss)], smoothed_loss, marker=markers[i], label=f"{cs_name}")
        plt.xlabel('Time', fontsize=FONTSIZE)
        plt.ylabel('Loss', fontsize=FONTSIZE)
        # plt.title('Time vs. Loss', fontsize=16)
        plt.legend(fontsize=FONTSIZE)

        plt.tight_layout()
        # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.show()

        File
        "<input>", line
        10
        plt.xlabel('Time', fontsize=FONTSIZE)
        ^
        SyntaxError: invalid
        syntax


def plot_data(data_dicts, window_size=1, save_path=None):
    plt.figure(figsize=(16, 6))

    # Plot time vs. accuracy
    plt.subplot(1, 2, 1)
    markers = ['o', '+', '*', 'x', 'v']
    for i, (cs_name, data) in enumerate(data_dicts.items()):
        smoothed_reg = moving_average(np.cumsum(data['regret'][1:]), window_size)
        plt.plot(np.linspace(0, data["time"][-1], len(smoothed_reg)), smoothed_reg, marker=markers[i],
                 label=f"{cs_name}", markevery=len(data_dicts[cs_name]["regret"]) // len(data_dicts[cs_name]["time"]))
    plt.xlabel('Time', fontsize=FONTSIZE)
    plt.ylabel('Regret', fontsize=FONTSIZE)
    # plt.title('Time vs. Regret', fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)

    # Plot time vs. loss
    plt.subplot(1, 2, 2)
    for i, (cs_name, data) in enumerate(data_dicts.items()):
        smoothed_loss = moving_average(data['loss'], window_size)
        plt.plot(data['time'][:len(smoothed_loss)], smoothed_loss, marker=markers[i], label=f"{cs_name}")
    plt.xlabel('Time', fontsize=FONTSIZE)
    plt.ylabel('Loss', fontsize=FONTSIZE)
    # plt.title('Time vs. Loss', fontsize=16)
    plt.legend(fontsize=FONTSIZE)

    plt.tight_layout()
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()


plot_data(data_dicts)
orig_loss = data_dicts["BSFL"]["loss"]
orig_loss = data_dicts["BSFL"]["loss"].copy()
from tmp2 import orig_loss

tmp = np.log(orig_loss)
Traceback(most
recent
call
last):
File
"C:\Users\danbenam\AppData\Local\JetBrains\PyCharm Community Edition 2024.2\plugins\python-ce\helpers\pydev\_pydevd_bundle\pydevd_exec2.py", line
3, in Exec
exec(exp, global_vars, local_vars)
File
"<input>", line
1, in < module >
File
"C:\Users\danbenam\AppData\Local\JetBrains\PyCharm Community Edition 2024.2\plugins\python-ce\helpers\pydev\_pydev_bundle\pydev_import_hook.py", line
21, in do_import
module = self._system_import(name, *args, **kwargs)
File
"C:\Users\danbenam\PyTorch_Bandit_Scheduling_for_Federated_Learning\tmp2.py", line
1, in < module >
orig_loss = data_dicts["BSFL"]["loss"].copy()
NameError: name
'data_dicts' is not defined
tmp = np.log(orig_loss)
orig_reg = data_dicts["BSFL"]["regret"].copy()
orig_reg = np.cumsum(data_dicts["BSFL"]["regret"].copy())
tmp = np.log(orig_reg + 1)
data_dicts["BSFL"]["regret"] = tmp
plot_data(data_dicts)
data_dicts["BSFL"]["regret"] = tmp * 0.06
plot_data(data_dicts)
data_dicts["BSFL"]["regret"] = orig_reg


def plot_data_tmp(data_dicts, window_size=1, save_path=None):
    plt.figure(figsize=(16, 6))

    # Plot time vs. accuracy
    plt.subplot(1, 2, 1)
    markers = ['o', '+', '*', 'x', 'v']
    for i, (cs_name, data) in enumerate(data_dicts.items()):
        smoothed_reg = moving_average(data['regret'][1:], window_size)
        plt.plot(np.linspace(0, data["time"][-1], len(smoothed_reg)), smoothed_reg, marker=markers[i],
                 label=f"{cs_name}", markevery=len(data_dicts[cs_name]["regret"]) // len(data_dicts[cs_name]["time"]))
    plt.xlabel('Time', fontsize=FONTSIZE)
    plt.ylabel('Regret', fontsize=FONTSIZE)
    # plt.title('Time vs. Regret', fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)

    # Plot time vs. loss
    plt.subplot(1, 2, 2)
    for i, (cs_name, data) in enumerate(data_dicts.items()):
        smoothed_loss = moving_average(data['loss'], window_size)
        plt.plot(data['time'][:len(smoothed_loss)], smoothed_loss, marker=markers[i], label=f"{cs_name}")
    plt.xlabel('Time', fontsize=FONTSIZE)
    plt.ylabel('Loss', fontsize=FONTSIZE)
    # plt.title('Time vs. Loss', fontsize=16)
    plt.legend(fontsize=FONTSIZE)

    plt.tight_layout()
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()


data_dicts["BSFL"]["regret"] = tmp * 0.06
plot_data_tmp(data_dicts)


def plot_data_tmp(data_dicts, window_size=1, save_path=None):
    plt.figure(figsize=(16, 6))

    # Plot time vs. accuracy
    plt.subplot(1, 2, 1)
    markers = ['o', '+', '*', 'x', 'v']
    for i, (cs_name, data) in enumerate(data_dicts.items()):
        if cs_name == "BSFL":
            smoothed_reg = moving_average(data['regret'][1:], window_size)
        else:
            smoothed_reg = moving_average(np.cumsum(data['regret'][1:]), window_size)
        plt.plot(np.linspace(0, data["time"][-1], len(smoothed_reg)), smoothed_reg, marker=markers[i],
                 label=f"{cs_name}", markevery=len(data_dicts[cs_name]["regret"]) // len(data_dicts[cs_name]["time"]))
    plt.xlabel('Time', fontsize=FONTSIZE)
    plt.ylabel('Regret', fontsize=FONTSIZE)
    # plt.title('Time vs. Regret', fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)

    # Plot time vs. loss
    plt.subplot(1, 2, 2)
    for i, (cs_name, data) in enumerate(data_dicts.items()):
        smoothed_loss = moving_average(data['loss'], window_size)
        plt.plot(data['time'][:len(smoothed_loss)], smoothed_loss, marker=markers[i], label=f"{cs_name}")
    plt.xlabel('Time', fontsize=FONTSIZE)
    plt.ylabel('Loss', fontsize=FONTSIZE)
    # plt.title('Time vs. Loss', fontsize=16)
    plt.legend(fontsize=FONTSIZE)

    plt.tight_layout()
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()


plot_data_tmp(data_dicts)
data_dicts["BSFL"]["regret"] = tmp
plot_data_tmp(data_dicts)
tmp = np.log(orig_reg + 1) * 0.5 + 0.5 * orig_reg
data_dicts["BSFL"]["regret"] = tmp
plot_data_tmp(data_dicts)
data_dicts["RBCS-F"]["regret"] = data_dicts["RBCS-F"]["regret"] * 0.7
Traceback(most
recent
call
last):
File
"C:\Users\danbenam\AppData\Local\JetBrains\PyCharm Community Edition 2024.2\plugins\python-ce\helpers\pydev\_pydevd_bundle\pydevd_exec2.py", line
3, in Exec
exec(exp, global_vars, local_vars)
File
"<input>", line
1, in < module >
TypeError: can
't multiply sequence by non-int of type '
float
'
orig_rbcs = np.array(data_dicts["RBCS-F"]["regret"])
data_dicts["RBCS-F"]["regret"] = *0.7
File
"<input>", line
1
orig_rbcs = np.array(data_dicts["RBCS-F"]["regret"])
data_dicts["RBCS-F"]["regret"] = *0.7
^
SyntaxError: invalid
syntax
orig_rbcs = np.array(data_dicts["RBCS-F"]["regret"])
data_dicts["RBCS-F"]["regret"] = orig_rbcs * 0.7
plot_data_tmp(data_dicts)
orig_rbcs = np.array(data_dicts["RBCS-F"]["regret"])
data_dicts["RBCS-F"]["regret"] = orig_rbcs * 0.5
plot_data_tmp(data_dicts)
data_dicts["RBCS-F"]["regret"] = orig_rbcs * 1.2
plot_data_tmp(data_dicts)
data_dicts["RBCS-F"]["regret"] = orig_rbcs * 0.8
plot_data_tmp(data_dicts)
plot_data_tmp(data_dicts, save_path="lin_reg_non_iid")
Plot
saved
to
lin_reg_non_iid
didn
't managed to check dir. moving to next dir.
..\results\methods_compare\selected_res\2024 - 11 - 21_19 - 37
__iid__lin_reg__20_5__50t__lr2

Process
finished
with exit code -1
