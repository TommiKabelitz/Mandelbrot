# Mandelbrot
Repository for visualising the Mandelbrot and Julia sets

# Requirements
 - Will run with the modules in minimal_requirements.txt
    - No adjustment in 
 - To run with MPI, require
    - a working MPI implementation supporting MPI-3
        - eg. OpenMPI
    - mpi4py (included in mpi_requirements.txt)
 - Python modules may be as usual installed with
 `pip install -r {requirements_file}`

# Usage
 - May be called on 1 process using
`python3 run.py mandelbrot`
or
`python3 run.py julia -c {real_c} {imag_c}`.
 - The c value is optional in both cases and will be randomised for the Julia set if not provided.
 - If running on multiple processes, simply prepend the above with the appropriate MPI syntax
    - eg `mpiexec -np {num_proc}` with OpenMPI




