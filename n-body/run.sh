#!/bin/bash
set -e

# all modes' names must be part of one git branch!
modes=('n-squared' 'serialized-trees' 'tree-building' 'merge-shared-trees')

STEPS=1000

# if the --build flag is passed to the script, pull and rebuild all implementations
if [[ "$1" == "--build" ]]; then
  for mode in "${modes[@]}"; do
    branch=$(git branch -l | grep $mode)
    echo "Checking out branch $branch..."
    git checkout $branch
    git pull

    echo "Building executable..."
    cargo build --release --target-dir ./target/$mode/
  done
fi

for n in 20 50 1000 5000 10000; do
  echo "N = $n, Steps = $STEPS..."

  for mode in "${modes[@]}"; do
    echo "Benchmarking model '$mode'..."

    for p in 1 2 4 8 16 28 40 80; do
      printf "$p processors: "

      # run the program, but only extract the running time info
      time=$(mpirun \
        -machinefile Machinefile \
        -mca btl_tcp_if_include eth0 \
        -np $p \
        ./target/$mode/release/n-body -n $n -s $STEPS \
        | grep 'It took' | cut -d' ' -f 3)
      printf "$time seconds\n"
    done

    echo ;
  done;
done;

