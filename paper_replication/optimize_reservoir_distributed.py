import os
import json
import subprocess
import argparse
import copy
import re
import math
import time


argParser = argparse.ArgumentParser()
argParser.add_argument("name", nargs="?", help="Name of this run, for logging, model saving, etc.", type=str)
#argParser.add_argument("-c", help="Location of config file that this run uses")
argParser.add_argument("-t", "--no-train", help="Don't train a new model, instead load the existing model.", action=argparse.BooleanOptionalAction)
argParser.add_argument("-T", "--tune", help="Tune the model with hyperband.", action=argparse.BooleanOptionalAction)
argParser.add_argument("-g", "--goal", help="Goal variable")
args = argParser.parse_args()

assert args.name
save_name = args.name
goal = args.goal
no_train_arg = False
tune_arg = False
if args.no_train:
    no_train_arg = '-t'
if args.tune:
    tune_arg = '-T'
if goal:
    goal = '-g'


hyperopt_config = {

    "exp": f"/stash/tlab/theom_intern/distributed_reservoir_runs/{save_name}",    # the experimentation name
    "hp_max_evals": 5,              # the number of differents sets of parameters hyperopt has to try
    "hp_method": "random",            # the method used by hyperopt to chose those sets (see below)
    "seed": 42,                       # the random state seed, to ensure reproducibility
    "instances_per_trial": 2,         # how many characteristics random ESN will be tried with each sets of parameters
    "hp_space": {                     # what are the ranges of parameters explored
        "N": ["choice", 5],             # the number of neurons is fixed to 500
        "sr": ["loguniform", 1e-2, 10],   # the spectral radius is log-uniformly distributed between 1e-2 and 10
        "lr": ["loguniform", .1, 1],    # idem with the leaking rate, from 1e-3 to 1
        "input_scaling": ["choice", 1.0], # the input scaling is fixed
        "ridge": ["loguniform", 1e-8, 1e1],        # and so is the regularization parameter.
        "seed": ["choice", 1234]          # an other random seed for the ESN initialization
    }
}

try: # try to make the directory that will store the runs with this config file, but if it is already made, we know the config
    # file is already made as well and we don't write to it
    os.mkdir(f"/stash/tlab/theom_intern/distributed_reservoir_runs/{save_name}")
    with open(f"/stash/tlab/theom_intern/distributed_reservoir_runs/{save_name}/{save_name}.config.json", "w+") as f:
        json.dump(hyperopt_config, f)

except FileExistsError:
    pass


def find_config_exponent_increment(parameter, param_min, param_max, num_cpus):
    # returns a dictionary with the key being the name of the variable parameter and the value being the range of the parameter divided by the number of cpus
    config_exponent_increment = {}
    config_exponent_increment[parameter] = math.log10(param_max / param_min) / num_cpus
    
    return config_exponent_increment

def apply_config_ranges(parameter, config_dict, increment, cpu, param_min):
    # returns a config dictionary with a different range of the specificed parameter
    new_config_dict = config_dict.copy()
    new_config_dict['hp_space'][parameter][2] = 10**(math.log10(param_min) + (cpu + 1)*increment[parameter])
    new_config_dict['hp_space'][parameter][1] = 10**(math.log10(param_min) + (cpu)*increment[parameter])
    new_config_dict['cpu'] = cpu
    new_config_dict['hp_space']['seed'][1] += cpu
    
    return new_config_dict

def create_configs(num_cpus, base_config_path, variable_parameter): # variable_parameter is the parameter that differs in range 
    # returns a list of config dictionaries with the range of the parameter argument different but every other parameter the same
    base_config = json.load(open(base_config_path))
    param_min = base_config['hp_space'][variable_parameter][1]
    param_max = base_config['hp_space'][variable_parameter][2]
    config_increment = find_config_exponent_increment(variable_parameter, param_min, param_max, num_cpus)

    configs = []
    for cpu in range(num_cpus):
        config = apply_config_ranges(variable_parameter, base_config, config_increment, cpu, param_min)
        configs.append(copy.deepcopy(config))
    
    return configs

def gather_cpus(cpus_to_search):
    cpus = []
    for cpu in cpus_to_search:
        cpu_data = subprocess.run('ssh ' + cpu + ' w', shell=True, text=True, capture_output=True).stdout
        load_avg_string = re.search(r' load average: \d+.\d+', cpu_data)
        load_avg = float(load_avg_string.group().split()[2])
        if load_avg <= 3.0:
            cpus.append(cpu)
    
    return cpus

def correlate_cpus_and_configs(variable_parameter):
    # gets a list of unused cpus, creates different config files for each, then returns dictionary of each server and their
    # config file
    cpus_to_search = ['rhone', 'saane',
                       'thur', 'ticino']
    cpus = gather_cpus(cpus_to_search)
    #cpus = ['arve', 'birs', 'inn', 'kander']
    num_cpus = len(cpus)
    
    parameter = variable_parameter
    configs = create_configs(num_cpus, f"/stash/tlab/theom_intern/distributed_reservoir_runs/{save_name}/{save_name}.config.json", parameter)
    cpus_and_configs = {}
    
    for i, cpu in enumerate(cpus):
        os.mkdir(f"/stash/tlab/theom_intern/distributed_reservoir_runs/{save_name}/{cpu}_hp_search")
        with open(f"/stash/tlab/theom_intern/distributed_reservoir_runs/{save_name}/{cpu}_hp_search/{cpu}.config.json", "w+") as f:
            json.dump(configs[i], f)

        cpus_and_configs[cpu] = configs[i]
    
    return cpus_and_configs

def a(word):
    if word:
        return str(word)
    else:
        return ''

def run_file_on_cpu(cpu_name, file_path, session_name, terminal_args): # file path is the whole file path from home
    # does not return anything
    subprocess.run('tmux new-window -S -n ' + cpu_name + '_window', shell=True)
    
    commands_t = ["cd ~/music_phrasing/tests/word_based",
                "python3 -m pipenv shell",
                # ". /u/theom_intern/.local/share/virtualenvs/word_based-6MIe0CKq/bin/activate",
                "conda activate tf",
                "cd ../../paper_replication/"]
    commands_b = ["conda activate music_phrasing_env"]

    if os.getlogin() == "theom_intern":
        commands = commands_t
    elif os.getlogin() == "brianl_intern":
        commands = commands_b
    else:
        raise ZeroDivisionError


    ssh_into_cpu_command = f'tmux send-keys -t {session_name} -l "ssh {cpu_name}"'
    #set_up_conda_env_command = f'tmux send-keys -t {session_name} -l "conda activate music_phrasing_env"'
    run_file_command = f'tmux send-keys -t {session_name} -l "python3 {file_path} {terminal_args.name} {cpu_name} {a(no_train_arg)} {a(tune_arg)} {a(goal)}"'
    enter_command = f'tmux send-keys -t {session_name} "Enter"'
     
    subprocess.run(ssh_into_cpu_command, shell=True)
    subprocess.run(enter_command, shell=True)
    time.sleep(1)
    for command in commands:
        subprocess.run(f'tmux send-keys -t {session_name} -l "{command}"', shell=True)
        subprocess.run(enter_command, shell=True)
        print(f"{session_name}: {command}")
        time.sleep(.5)
        if command == "python3 -m pipenv shell":
            time.sleep(4)
    subprocess.run(run_file_command, shell=True)
    subprocess.run(enter_command, shell=True)

def run_file_on_all_cpus(cpus, hp_optimization_file_path, tmux_session_name, terminal_args): # hp_optimization_file_path is the path to the
    # file we are running as a string
    initialize_session_command = f'tmux new-session -d -s {tmux_session_name}'
    subprocess.run(initialize_session_command, shell=True)
    for cpu in list(cpus):
        run_file_on_cpu(cpu, hp_optimization_file_path, tmux_session_name, terminal_args)
        time.sleep(2)


def hp_optimization_parallelized(hp_optimization_file_path, tmux_session_name, terminal_args): # terminal args is a Namespace
    # object returned by the parse_args function
    # main tuning function, should replace(and call) research in the if args.tune statement
    # all other hyper parameter optimization parallelization functions should feed into this one
    cpus_and_configs = correlate_cpus_and_configs('lr')
    cpus = cpus_and_configs.keys()

    run_file_on_all_cpus(cpus, hp_optimization_file_path, tmux_session_name, terminal_args)


hp_optimization_parallelized('~/Downloads/music_phrasing/paper_replication/optimize_reservoir.py' if os.getlogin() == "brianl_intern" else 'optimize_reservoir.py', f"theobrian_{save_name}", args)
