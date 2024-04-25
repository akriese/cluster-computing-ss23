# Rust and MPI

This solution is using the rsmpi crate to interface with the C libraries for MPI.

Build the project with `cargo build --release` and run it by executing `mpirun -np $NUM_PROCESSORS ./target/release/u1`.

This assumes, that MPICH was installed previously.

## Input

Optionally, one can pass a command line argument to the program execution pointing to a
json file. This file contains two keys "a" and "b" whose values are matrices.
The program is then run with: `mpirun -np $NUM_PROCESSORS ./target/release/u1 $INPUT_FILE_NAME`
