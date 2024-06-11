# Usage

Build with `cargo build --release`

Run with `mpirun -N $NUM_NODES -np $NUM_NODES ./target/release/n-body -t $THREADS_PER_NODE -n $NUM_BODIES -s $NUM_STEPS`
