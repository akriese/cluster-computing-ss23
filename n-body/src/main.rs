mod tree;

use clap::Parser;
use mpi::traits::*;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    iter::repeat,
    time::{Duration, Instant},
};
use tree::TreeNode;

const G: f64 = 6.67e-11f64;
const ROOT_RANK: usize = 0;

#[derive(Parser, Debug)]
#[command(version, about, long_about=None)]
struct Args {
    #[arg(short = 'M', default_value_t = 1e3f64)]
    mass_max: f64,

    #[arg(short = 'P', default_value_t = 1e2f64)]
    pos_max: f64,

    #[arg(short = 'S', default_value_t = 1e0f64)]
    velocity_max: f64,

    #[arg(short = 'n', default_value_t = 1000)]
    n_bodies: usize,

    #[arg(short = 's', default_value_t = 1000)]
    n_steps: usize,

    #[arg(short = 'l', default_value_t = 0.1)]
    step_time: f64,

    #[arg(short = 'p', action)]
    print: bool,

    #[arg(short = 'T', default_value_t = 0.5)]
    theta: f64,

    #[arg(short = 't')]
    threads_per_node: Option<usize>,
}

#[derive(Clone, Debug, Default, Equivalence, Deserialize, Serialize)]
struct Body {
    id: usize,
    mass: f64,
    position: [f64; 2],
    velocity: [f64; 2],
}

/// Generates a float vector of the given length within a given min-max range.
///
/// * `n`: Length of the output vector.
/// * `min`: Minimum of the generated values.
/// * `max`: Maximum of the generated values.
fn generate_random_bounded(n: usize, min: f64, max: f64) -> Vec<f64> {
    let mut result = vec![0f64; n];
    thread_rng().fill(&mut result[..]);

    result.iter().map(|x| x * (max - min) + min).collect()
}

/// Gather outer bounds of all given bodies
///
/// * `positions`: Positions of all bodies.
fn get_bounds(positions: &[[f64; 2]]) -> [[f64; 2]; 2] {
    [
        [
            positions
                .iter()
                .map(|p| p[0])
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
            positions
                .iter()
                .map(|p| p[0])
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
        ],
        [
            positions
                .iter()
                .map(|p| p[1])
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
            positions
                .iter()
                .map(|p| p[1])
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
        ],
    ]
}

static mut SUBTREE_DURATIONS: Vec<Duration> = vec![];
static mut MERGE_DURATIONS: Vec<Duration> = vec![];
static mut CALC_DURATIONS: Vec<Duration> = vec![];
static mut GATHER_DURATIONS: Vec<Duration> = vec![];

/// Execute one parallelized step of the Barnes-Hut algorithm.
///
/// 1. Create one subtree per thread in parallel.
/// 2. Merge the trees into one big tree.
/// 3. Calculate forces recursively for the local bodies in parallel.
///
/// * `timestep`: Size of timesteps
/// * `theta`: Theta threshold of the algorithm
/// * `all_bodies`: All bodies of the system.
/// * `local_bodies`: Bodies to compute values for.
/// * `root`: Root tree node which already contains size and center respecting ALL bodies.
/// * `n_threads`: Number of parallel threads available.
fn barnes_hut(
    timestep: f64,
    theta: f64,
    all_bodies: &mut Vec<Body>,
    root: &mut TreeNode,
    n_threads: usize,
    n_proc: usize,
    rank: usize,
) {
    let mut start_time = std::time::Instant::now();

    let bodies_per_thread = all_bodies.len() / n_threads;

    // create NUM_THREADS trees in parallel
    let mut thread_trees = all_bodies
        .par_chunks(bodies_per_thread)
        .map(|bs| {
            let mut thread_root = root.clone();

            bs.iter().for_each(|b| {
                if b.mass > 0f64 {
                    thread_root.insert(b);
                }
            });

            thread_root
        })
        .collect::<Vec<TreeNode>>();

    unsafe {
        SUBTREE_DURATIONS.push(start_time.elapsed());
    }
    start_time = Instant::now();

    println!("merging...");

    // merge the trees to a big tree, merge the trees in parallel
    thread_trees
        .drain(..)
        .par_bridge()
        .reduce_with(|mut a, b| {
            a.merge(b);
            a
        })
        .unwrap();
    println!("done...");

    unsafe {
        MERGE_DURATIONS.push(start_time.elapsed());
    }
    start_time = Instant::now();

    // calculate forces, velocity and positions
    all_bodies[rank * bodies_per_thread..(rank + 1) * bodies_per_thread]
        .par_iter_mut()
        .for_each(|b| {
            if b.mass == 0f64 {
                return;
            }

            let f = root.calculate_force(b, theta);
            b.velocity = calc_velocity(&b.velocity, &f, b.mass, timestep);
            b.position = calc_position(&b.velocity, &b.position, timestep);
        });

    unsafe {
        CALC_DURATIONS.push(start_time.elapsed());
    }
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let n_nodes = world.size();
    let rank = world.rank() as usize;

    let is_root = rank == ROOT_RANK;

    // parse hyperparameteres; shared between all processes without sending them actively
    let args = Args::parse();

    if let Some(threads) = args.threads_per_node {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }

    let n_threads = args.threads_per_node.unwrap_or(1);

    let start_time = Instant::now();

    // we add zero weight bodies at the end
    // so that all processes get the same amount of bodies
    let bodies_per_proc = (args.n_bodies as f64 / n_nodes as f64).ceil() as usize;
    let filled_n = bodies_per_proc * n_nodes as usize;
    let extra_n = filled_n - args.n_bodies;

    let mut all_bodies = vec![Body::default(); filled_n];

    // create initial bodies
    if is_root {
        let mut masses = generate_random_bounded(args.n_bodies, 0f64, args.mass_max);
        masses.extend(repeat(0f64).take(extra_n));

        let mut all_positions =
            generate_random_bounded(args.n_bodies * 2, -args.pos_max, args.pos_max);
        all_positions.extend(repeat(0f64).take(extra_n * 2));

        let mut init_velocities =
            generate_random_bounded(args.n_bodies * 2, -args.velocity_max, args.velocity_max);
        init_velocities.extend(repeat(0f64).take(extra_n * 2));

        for (i, b) in all_bodies.iter_mut().enumerate() {
            b.id = i;
            b.mass = masses[i];
            b.position = all_positions[i * 2..(i + 1) * 2].try_into().unwrap();
            b.velocity = init_velocities[i * 2..(i + 1) * 2].try_into().unwrap();
        }
    }

    // share all bodies with all processes
    world
        .process_at_rank(ROOT_RANK as i32)
        .broadcast_into(&mut all_bodies);

    for _step in 0..args.n_steps {
        // initial tree root
        let bounds = get_bounds(
            &all_bodies
                .iter()
                .map(|b| b.position)
                .collect::<Vec<[f64; 2]>>(),
        );
        let size = f64::max(bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0]);
        let mut tree = TreeNode {
            center: [
                (bounds[0][1] + bounds[0][0]) / 2f64,
                (bounds[1][1] + bounds[1][0]) / 2f64,
            ],
            size,
            ..TreeNode::default()
        };

        barnes_hut(
            args.step_time,
            args.theta,
            &mut all_bodies,
            &mut tree,
            n_threads,
            world.size() as usize,
            rank,
        );

        let start_time = Instant::now();
        // println!("Rank {} entered the broadcast at {:?}", rank, start_time);
        // share bodies between all processes
        for r in 0..world.size() as usize {
            world
                .process_at_rank(r as i32)
                .broadcast_into(&mut all_bodies[r * bodies_per_proc..(r + 1) * bodies_per_proc]);
        }
        // world.all_gather_into(&local_bodies, &mut all_bodies);
        unsafe { GATHER_DURATIONS.push(start_time.elapsed()) }
        // println!(
        //     "Rank {} finished the Allgather at {:?}",
        //     rank,
        //     Instant::now()
        // );
    }

    println!(
        "Rank {}: Avg subtree building duration: {:.2?}",
        rank,
        avg_duration(unsafe { &SUBTREE_DURATIONS })
    );
    println!(
        "Rank {}: Avg merge duration: {:.2?}",
        rank,
        avg_duration(unsafe { &MERGE_DURATIONS })
    );
    println!(
        "Rank {}: Avg force calc duration: {:.2?}",
        rank,
        avg_duration(unsafe { &CALC_DURATIONS })
    );
    println!(
        "Rank {}: Avg gather duration: {:.2?}",
        rank,
        avg_duration(unsafe { &GATHER_DURATIONS })
    );

    if is_root {
        println!("It took {:.2?}!", start_time.elapsed());
    }
}

/// Calculate the average duration from a list of durations.
///
/// * `durations`: Durations to calculate the avg for.
fn avg_duration(durations: &[Duration]) -> Duration {
    assert!(!durations.is_empty());

    let mut summed_durs = durations[0];

    durations
        .iter()
        .skip(1)
        .for_each(|d| summed_durs = summed_durs.checked_add(*d).unwrap());

    summed_durs / durations.len() as u32
}

/// Calculate the new velocity of a body.
///
/// * `old_velocity`: Old velocity
/// * `force`: Current force on the body
/// * `mass`: Body's mass
/// * `timestep`: Step size of the time
fn calc_velocity(old_velocity: &[f64; 2], force: &[f64; 2], mass: f64, timestep: f64) -> [f64; 2] {
    let [v_x, v_y] = old_velocity;
    let [f_x, f_y] = force;
    [v_x + f_x / mass * timestep, v_y + f_y / mass * timestep]
}

/// Calculate the new position of a body.
///
/// * `velocity`: New velocity
/// * `old_position`: Old position
/// * `timestep`: Time step size
fn calc_position(velocity: &[f64; 2], old_position: &[f64; 2], timestep: f64) -> [f64; 2] {
    let [v_x, v_y] = velocity;
    let [x, y] = old_position;
    [x + v_x * timestep, y + v_y * timestep]
}
