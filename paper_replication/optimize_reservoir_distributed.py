import json
import subprocess
import argparse
import copy
import re

from reservoirpy.hyper import research


argParser = argparse.ArgumentParser()
argParser.add_argument("name", nargs="?", help="Name of this run, for logging, model saving, etc.", type=str)
argParser.add_argument("-t", "--no-train", help="Don't train a new model, instead load the existing model.", action=argparse.BooleanOptionalAction)
argParser.add_argument("-T", "--tune", help="Tune the model with hyperband.", action=argparse.BooleanOptionalAction)
argParser.add_argument("-g", "--goal", help="Goal variable")
args = argParser.parse_args()

assert args.name
save_name = args.name
goal = args.goal
if not goal:
    goal = "Micro"
assert goal in ["Micro", "Len_P"]


def find_config_increments(parameter, config_dict, num_cpus):
    # returns a dictionary with the key being the name of the variable parameter and the value being the range of the parameter divided by the number of cpus
    config_increments = {}
    config_increments[parameter] = (config_dict['hp_space'][parameter][2] - config_dict['hp_space'][parameter][1]) / num_cpus
    
    return config_increments

def apply_config_ranges(parameter, config_dict, increments, cpu):
    # returns a config dictionary with a different range of the specificed parameter
    new_config_dict = config_dict.copy()
    new_config_dict['hp_space'][parameter][2] = (cpu + 1) * increments[parameter]
    new_config_dict['hp_space'][parameter][1] = (cpu) * increments[parameter]
    new_config_dict['cpu'] = cpu
    
    return new_config_dict

def create_configs(num_cpus, base_config_path, variable_parameter): # variable_parameter is the parameter that differs in range 
    # returns a list of config dictionaries with the range of the parameter argument different but every other parameter the same
    base_config = json.load(open(base_config_path))
    config_increments = find_config_increments(variable_parameter, base_config, num_cpus)

    configs = []
    for cpu in range(num_cpus):
        config = apply_config_ranges(variable_parameter, base_config, config_increments, cpu)
        configs.append(copy.deepcopy(config))
    
    return configs

def gather_cpus(cpus_to_search):
    cpus = []
    for cpu in cpus_to_search:
        cpu_data = subprocess.run('sosh ' + cpu + ' w', shell=True).stdout
        load_avg_string = re.search(' load average: \d+.\d+', cpu_data)
        load_avg = float(load_avg_string.group().split()[2])
        if load_avg <= 3.0:
            cpus.append(cpu)
    
    return cpus

def correlate_cpus_and_configs(variable_parameter):
    # gets a list of unused cpus, creates different config files for each, then returns dictionary of each server and their
    # config file
    cpus_to_search = ['arve', 'birs', 'doubs', 'inn', 'kander', 'linth', 'lonza', 'orbe', 'reuss', 'rhine', 'rhone', 'saane',
                       'thur', 'ticino']
    cpus = gather_cpus(cpus_to_search)
    num_cpus = len(cpus)
    
    parameter = variable_parameter
    configs = create_configs(num_cpus, "/stash/tlab/theom_intern/hp_model_configs/search_space.json", parameter)
    cpus_and_configs = {}
    
    for i, cpu in enumerate(cpus):
        cpus_and_configs[cpu] = configs[i]
    
    return cpus_and_configs

def run_file_on_cpu(cpu_name, file_path, session_name, terminal_args): # file path is the whole file path from home
    # does not return anything
    subprocess.run('tmux new-window -S -n ' + cpu_name + '_window', shell=True)

    set_up_file_command = f'tmux send-keys -t {session_name} -l "ssh {cpu_name} python {file_path} {terminal_args.name} {terminal_args.no_train} {terminal_args.tune} {terminal_args.goal}"'
    run_file_command = f'tmux send-keys -t {session_name} "Enter"'
    
    subprocess.run(set_up_file_command, shell=True)
    subprocess.run(run_file_command, shell=True)    

def run_file_on_all_cpus(cpus, hp_optimization_file_path, tmux_session_name, terminal_args): # hp_optimization_file_path is the path to the
    # file we are running as a string
    initialize_session_command = f'tmux new-session -d -s {tmux_session_name}'
    subprocess.run(initialize_session_command, shell=True)
    for cpu in list(cpus):
        run_file_on_cpu(cpu, hp_optimization_file_path, tmux_session_name, terminal_args)


def hp_optimization_parallelized(hp_optimization_file_path, tmux_session_name, terminal_args): # terminal args is a Namespace
    # object returned by the parse_args function
    # main tuning function, should replace(and call) research in the if args.tune statement
    # all other hyper parameter optimization parallelization functions should feed into this one
    cpus_and_configs = correlate_cpus_and_configs('lr')
    cpus = cpus_and_configs.keys()

    run_file_on_all_cpus(cpus, hp_optimization_file_path, tmux_session_name, terminal_args)


hp_optimization_parallelized('~/Downloads/music_phrasing/paper_replication/optimize_reservoir.py', 'test_session')
