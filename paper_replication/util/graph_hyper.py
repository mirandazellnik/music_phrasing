from reservoirpy.hyper import plot_hyperopt_report # type: ignore
import argparse


argParser = argparse.ArgumentParser()
argParser.add_argument("name", nargs="?", help="Name of the run to plot", type=str)
argParser.add_argument("arg_num", nargs="?", help="Number of args that were optimized (5 or 3)", type=int)
args = argParser.parse_args()

assert args.name
save_name = args.name
assert args.arg_num
num = args.arg_num

assert (num in [3, 5])
if (num == 3):
    args = ("lr", "sr", "ridge")
elif (num == 5):
    args = ("lr", "sr", "ridge", "rc_conectivity", "input_connectivity")
fig = plot_hyperopt_report(f"/stash/tlab/theom_intern/distributed_reservoir_runs/{save_name}", args, metric="r2")
fig.savefig(f"/stash/tlab/theom_intern/distributed_reservoir_runs/{save_name}/hyper_graph.png")
fig.show()