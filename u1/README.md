# Rust and MPI

This solution is using the rsmpi crate to interface with the C libraries for MPI.

Build the project with `cargo build --release` and run it by executing `mpirun -np $NUM_PROCESSORS ./target/release/u1`.

This assumes, that MPICH was installed previously.