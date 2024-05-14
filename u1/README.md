# Rust and MPI

This solution is using the rsmpi crate to interface with the C libraries for MPI.

Build the project with `cargo build --release` and run it by executing `mpirun -np $NUM_PROCESSORS ./target/release/u1`.

This assumes, that MPICH was installed previously (e.g. `sudo apt install mpich`).

On the FU compute cluster, the flag `-mca btl_tcp_if_include eth0` has to be included
as otherwise no connections to other nodes are possible.

A recommended bash function can be found in `run.sh`.

This loop will execute `run.sh` with the big matrix with different numbers of processes:
`for p in {1..100..5}; do printf "$x: "; ./run.sh $p $stride input.json 2> /dev/null | grep 'It took' | cut -d' ' -f3; done;`

## Input

Optionally, one can pass a command line argument to the program execution pointing to a
json file. This file contains two keys "a" and "b" whose values are matrices.
The program is then run with: `mpirun -np $NUM_PROCESSORS ./target/release/u1 $INPUT_FILE_NAME`
