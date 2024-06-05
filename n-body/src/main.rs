mod tree;

use clap::Parser;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{iter::repeat, time::Instant};
use tree::TreeNode;

const G: f64 = 6.67e-11f64;

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
    num_threads: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
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

/// Execute one parallelized step of the Barnes-Hut algorithm.
///
/// 1. Create a tree from the local bodies.
/// 2. Serialize the tree.
/// 3. Share tree with other processes and gather from them.
/// 4. Deserialize others' trees.
/// 5. Merge others' trees into own.
/// 6. Calculate forces recursively for the local bodies.
///
/// * `timestep`: Size of timesteps
/// * `theta`: Theta threshold of the algorithm
/// * `bodies`: Bodies to compute values for.
/// * `root`: Root tree node which already contains size and center respecting ALL bodies.
/// * `n_threads`: Number of parallel threads available.
fn barnes_hut(
    timestep: f64,
    theta: f64,
    bodies: &mut Vec<Body>,
    root: &mut TreeNode,
    n_threads: usize,
) {
    let mut start_time = std::time::Instant::now();

    let bodies_per_thread = bodies.len() / n_threads;

    // create NUM_THREADS trees in parallel
    let thread_trees = (0..n_threads)
        .into_par_iter()
        .map(|t| {
            let mut thread_root = root.clone();

            bodies[t * bodies_per_thread..(t + 1) * bodies_per_thread]
                .iter()
                .for_each(|b| {
                    if b.mass > 0f64 {
                        thread_root.insert(b);
                    }
                });

            thread_root
        })
        .collect::<Vec<TreeNode>>();

    println!(
        "subtrees built! time since step started: {:.2?}",
        start_time.elapsed()
    );
    start_time = Instant::now();

    // merge the trees to a big tree
    for tree in thread_trees {
        root.merge(tree);
    }

    println!(
        "Trees merged! time since step started: {:.2?}",
        start_time.elapsed()
    );
    start_time = Instant::now();

    // calculate forces, velocity and positions for given range
    bodies.par_iter_mut().for_each(|b| {
        if b.mass == 0f64 {
            return;
        }

        let f = root.calculate_force(b, theta);
        b.velocity = calc_velocity(&b.velocity, &f, b.mass, timestep);
        b.position = calc_position(&b.velocity, &b.position, timestep);
    });

    println!(
        "Forces calculated! time since step started: {:.2?}",
        start_time.elapsed()
    );
}

fn main() {
    // parse hyperparameteres; shared between all processes without sending them actively
    let args = Args::parse();

    if let Some(threads) = args.num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }

    let n_proc = args.num_threads.unwrap_or(1);

    let start_time = Instant::now();

    // we add zero weight bodies at the end
    // so that all processes get the same amount of bodies
    let bodies_per_proc = (args.n_bodies as f64 / n_proc as f64).ceil() as usize;
    let filled_n = bodies_per_proc * n_proc;
    let extra_n = filled_n - args.n_bodies;

    let mut all_bodies = vec![Body::default(); filled_n];

    // create input
    let mut masses = generate_random_bounded(args.n_bodies, 0f64, args.mass_max);
    masses.extend(repeat(0f64).take(extra_n));

    let mut all_positions = generate_random_bounded(args.n_bodies * 2, -args.pos_max, args.pos_max);
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
            args.num_threads.unwrap_or(1),
        );
    }

    println!("It took {:.2?}!", start_time.elapsed());
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
