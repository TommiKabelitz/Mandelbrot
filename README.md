# Mandelbrot
Repository for visualising the Mandelbrot and Julia sets

![out](https://user-images.githubusercontent.com/71644734/186583297-ef74e464-ecfd-42ee-8f67-ad48a1a58164.gif)

# Features
 - Customisable visualisation of Mandelbrot and Julia sets
 - Zoom in and out with left/right click
 - Optional MPI support (Runs much faster on multiple processes)
 - Customisable zoom rate
 - Jump to specified zoom level
 - Customisable colour mapping
 - Basic runscript provided
    - Simple interface if desired

# Requirements
 - Will run with the modules in minimal_requirements.txt
    - No change required
 - To run with MPI, require
    - a working MPI implementation supporting MPI-3
        - eg. OpenMPI
    - mpi4py (included in mpi_requirements.txt)
 - Python modules may be as usual installed with
 `pip install -r {requirements_file}`

# Usage
 - May be called without MPI using

`python3 run.py mandelbrot`

or

`python3 run.py julia -c {real_c} {imag_c}`.

 - The c value is optional in both cases and will be randomised for the Julia set if not provided.
 - If running on multiple processes, simply prepend the above with the appropriate MPI syntax
    - eg `mpiexec -np {num_proc}` with OpenMPI
 - Customisation possible through parameters.yml




