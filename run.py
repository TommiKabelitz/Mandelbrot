import argparse
from cProfile import run
import random
from typing import Any
import complex_sets as cs

from utilities import Put

def command_line_interface() -> dict:

    parser = argparse.ArgumentParser(description="Interactively plot complex_set or Julia sets. MPI is supported through mpi4py.")
    parser.add_argument("set_type",choices=("Mandelbrot","mandelbrot","Julia","julia"),help="Type of complex set to plot.")
    parser.add_argument("-c",nargs=2,type=float,required=False,help="c value to use when plotting Julia sets. Specify 2 floats, ordered real, then imaginary.")
    args = parser.parse_args()
    return vars(args)

def run_set(complex_set: Any):
    complex_set.allocate_arrays()
    complex_set.initialise_arrays()
    complex_set.update_colour()
    complex_set.initialise_plot()
    complex_set.zoom_loop()


if __name__ == "__main__":

    inputArgs = command_line_interface()
    
    if inputArgs["set_type"].title() == "Julia":
        if inputArgs["c"] is not None:
            args = [complex(*inputArgs["c"])]
        else:
            Put("c value not specified for Julia set. Randomising c.",root=True)
            random.seed()
            args = [complex(random.random(),random.random())]
    else:
        args = []
    Put(args,root=True)

    complex_set = getattr(cs,inputArgs["set_type"].title())(*args)
    print(complex_set.c)
    run_set(complex_set)


