use mpi::traits::*;
use rand::{thread_rng, Rng};

const N_BODIES: usize = 4000;
const N_STEPS: usize = 5000;
const ROOT_RANK: usize = 0;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let root_proc = world.process_at_rank(ROOT_RANK as i32);
    let n_proc = world.size() as usize;
    let rank = world.rank() as usize;

    let start_time = mpi::time();

    // we add zero weight bodies at the end
    // so that all processes get the same amount of bodies
    let filled_n = ((N_BODIES as f64 / n_proc as f64).ceil() as usize) * n_proc;

    let mut masses: Vec<f64> = vec![0f64; filled_n];
    let mut all_positions: Vec<f64> = vec![0f64; filled_n * 2];

    // root creates input
    if rank == ROOT_RANK {
        thread_rng().fill(&mut masses[..N_BODIES]);

        // dummy values
        all_positions = (0..filled_n * 2).map(|x| x as f64).collect::<Vec<f64>>();
        // use this otherwise for random values:
        // thread_rng().fill(&mut all_positions[..N_BODIES*2]);
    }

    // root sends masses
    root_proc.broadcast_into(&mut masses);

    let n_bodies = masses.len();
    let bodies_per_proc = n_bodies / n_proc as usize;

    // root sends initial coordinates to everyone
    root_proc.broadcast_into(&mut all_positions);

    // root sends initial velocity to respective ranks
    let mut local_velocities = vec![0f64; bodies_per_proc * 2];
    if rank == ROOT_RANK {
        let mut init_velocities: Vec<f64> = vec![0f64; filled_n * 2];

        init_velocities = all_positions.clone();
        // use this for random values instead:
        // thread_rng().fill(&mut init_velocities[..N_BODIES * 2]);
        root_proc.scatter_into_root(&init_velocities, &mut local_velocities);
    } else {
        root_proc.scatter_into(&mut local_velocities);
    }

    // println!(
    //     "Process {} got initial positions {:?} and initial velocities {:?}",
    //     rank, all_positions, local_velocities
    // );

    let mut local_positions =
        all_positions[rank * bodies_per_proc * 2..(rank + 1) * bodies_per_proc * 2].to_vec();

    for t in 0..N_STEPS {
        // calculate their velocity and positions
        (local_positions, local_velocities) =
            calculate_positions(&masses, &local_velocities, &local_positions, &all_positions);

        // send new positions with MPI_Allgather
        world.all_gather_into(&local_positions, &mut all_positions);
        world.barrier();

        // println!(
        //     "Process {} at step {} got positions {:?}",
        //     rank, t, all_positions
        // );
    }

    if rank == ROOT_RANK {
        println!("It took {} seconds!", mpi::time() - start_time);
    }
}

fn calculate_positions(
    masses: &Vec<f64>,
    local_velocities: &Vec<f64>,
    local_positions: &Vec<f64>,
    all_positions: &Vec<f64>,
) -> (Vec<f64>, Vec<f64>) {
    return (
        local_positions
            .iter()
            .map(|x| x + 1f64)
            .collect::<Vec<f64>>(),
        local_velocities
            .iter()
            .map(|x| x * 2f64)
            .collect::<Vec<f64>>(),
    );
}
