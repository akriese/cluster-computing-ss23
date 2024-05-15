use itertools::Itertools;
use mpi::traits::*;
use rand::{thread_rng, Rng};

const N_BODIES: usize = 1000;
const N_STEPS: usize = 1000;
const ROOT_RANK: usize = 0;
const G: f64 = 6.67e-11f64;
const TIMESTEPS: f64 = 0.1;

fn generate_random_bounded(n: usize, min: f64, max: f64) -> Vec<f64> {
    let mut result = vec![0f64; n];
    thread_rng().fill(&mut result[..]);

    result.iter().map(|x| x * (max - min) + min).collect()
}

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
        masses = generate_random_bounded(filled_n, 5e2, 5e3);
        all_positions = generate_random_bounded(filled_n * 2, -1e1, 1e1);
    }

    // if rank == ROOT_RANK {
    //     println!("Masses: {:?}", masses);
    // }

    // root sends masses
    root_proc.broadcast_into(&mut masses);

    let n_bodies = masses.len();
    let bodies_per_proc = n_bodies / n_proc as usize;

    // root sends initial coordinates to everyone
    root_proc.broadcast_into(&mut all_positions);

    // if rank == ROOT_RANK {
    //     println!("{:?}", all_positions);
    // }

    // root sends initial velocity to respective ranks
    let mut local_velocities = vec![0f64; bodies_per_proc * 2];
    if rank == ROOT_RANK {
        let init_velocities = generate_random_bounded(filled_n * 2, -1e0, 1e0);
        // let init_velocities = vec![0f64; filled_n * 2];
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
        (local_positions, local_velocities) = calculate_next_step(
            &local_velocities,
            &local_positions,
            rank,
            &masses,
            &all_positions,
            TIMESTEPS,
        );

        // send new positions with MPI_Allgather
        world.all_gather_into(&local_positions, &mut all_positions);
        world.barrier();

        // if rank == ROOT_RANK {
        //     println!("{:?}", all_positions);
        // }
    }

    if rank == ROOT_RANK {
        println!("It took {} seconds!", mpi::time() - start_time);
    }
}

fn calculate_next_step(
    local_velocities: &Vec<f64>,
    local_positions: &Vec<f64>,
    local_offset: usize,
    masses: &Vec<f64>,
    all_positions: &Vec<f64>,
    timestep: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = local_positions.len();

    assert!(n % 2 == 0);

    let mut new_velocities = vec![0f64; n];
    let mut new_positions = vec![0f64; n];

    for i in 0..n / 2 {
        let range = i * 2..(i + 1) * 2;
        let p: &[f64; 2] = &local_positions[range.clone()].try_into().unwrap();
        let v: &[f64; 2] = &local_velocities[range.clone()].try_into().unwrap();
        let m = masses[local_offset + i];
        let f = calc_force(p, m, all_positions, masses);
        let new_v = calc_velocity(v, &f, m, timestep);
        let new_p = calc_position(&new_v, p, timestep);

        new_velocities[range.clone()].copy_from_slice(&new_v);
        new_positions[range.clone()].copy_from_slice(&new_p);
    }

    (new_positions, new_velocities)
}

fn calc_velocity(old_velocity: &[f64; 2], force: &[f64; 2], mass: f64, timestep: f64) -> [f64; 2] {
    let [v_x, v_y] = old_velocity;
    let [f_x, f_y] = force;
    return [v_x + f_x / mass * timestep, v_y + f_y / mass * timestep];
}

fn calc_position(velocity: &[f64; 2], old_position: &[f64; 2], timestep: f64) -> [f64; 2] {
    let [v_x, v_y] = velocity;
    let [x, y] = old_position;
    return [x + v_x * timestep, y + v_y * timestep];
}

fn calc_force(
    self_position: &[f64; 2],
    self_mass: f64,
    other_positions: &Vec<f64>,
    masses: &Vec<f64>,
) -> [f64; 2] {
    let mut summed_force = [0f64; 2];
    let [self_x, self_y] = self_position;
    for (i, (x, y)) in other_positions.iter().tuples().enumerate() {
        let (d_x, d_y) = (x - self_x, y - self_y);
        if d_x == 0f64 && d_y == 0f64 {
            continue;
        }
        let r = (d_x * d_x + d_y * d_y).sqrt();
        let f = G * self_mass * masses[i] / (r * r);
        summed_force[0] += f + d_x / r;
        summed_force[1] += f + d_y / r;
    }

    return summed_force;
}
