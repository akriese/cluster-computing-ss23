NUM_PROCESSES=$1

cargo build --release
mpirun -machinefile Machinefile -mca btl_tcp_if_include eth0 -np $NUM_PROCESSES ./target/release/u1
