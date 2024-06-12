# Usage

Build with `cargo build --release`

Run with `mpirun -N $NUM_NODES -np $NUM_NODES ./target/release/n-body -t $THREADS_PER_NODE -n $NUM_BODIES -s $NUM_STEPS`

It turned out, that using only one or two nodes would not properly make MPI bind
to the wanted amount of CPUs. This could be some hidden policy of the MPI implementation.
We found out about this by adding the `--report-bindings` flag.
To ensure, that MPI binds each process to an entire socket, one has to use the
flag `--bind-to socket` and `--map-by node`.

