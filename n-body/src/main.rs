mod tree;

use clap::Parser;
use mpi::datatype::PartitionMut;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::iter::repeat;
use tree::TreeNode;

const ROOT_RANK: usize = 0;
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

    #[arg(short = 't', default_value_t = 0.5)]
    theta: f64,
}

#[derive(Clone, Debug, Equivalence, Default, Deserialize, Serialize)]
struct Body {
    id: usize,
    mass: f64,
    position: [f64; 2],
    velocity: [f64; 2],
}

/// Calculate the center for the root node.
///
/// * `bodies`: Slice of all bodies.
fn calc_center(bodies: &[Body]) -> [f64; 2] {
    let x = bodies.iter().map(|b| b.position[0]).sum::<f64>() / bodies.len() as f64;
    let y = bodies.iter().map(|b| b.position[1]).sum::<f64>() / bodies.len() as f64;

    [x, y]
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

/// Gather outer bounds of all given bodies and get the max of both dimensions.
///
/// * `positions`: Positions of all bodies.
fn get_size(positions: &[[f64; 2]]) -> f64 {
    f64::max(
        positions
            .iter()
            .map(|p| p[0])
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            - positions
                .iter()
                .map(|p| p[0])
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
        positions
            .iter()
            .map(|p| p[1])
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            - positions
                .iter()
                .map(|p| p[1])
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
    )
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
/// * `world`: MPI communicator
/// * `timestep`: Size of timesteps
/// * `theta`: Theta threshold of the algorithm
/// * `local_bodies`: Bodies to compute values for locally.
/// * `root`: Root tree node which already contains size and center respecting ALL bodies.
fn barnes_hut(
    world: &SimpleCommunicator,
    timestep: f64,
    theta: f64,
    local_bodies: &mut Vec<Body>,
    root: &mut TreeNode,
) {
    let root_copy = root.clone();
    let mut start_time = mpi::time();
    let mut current_time;

    for body in local_bodies.iter() {
        if body.mass > 0f64 {
            root.insert(&body);
        }
    }

    if world.rank() == 0 as i32 {
        current_time = mpi::time();
        println!(
            "Tree built! time since step started: {} sec",
            current_time - start_time
        );
        start_time = current_time;
    }

    // serialize own tree
    let serialized = bitcode::serialize(&root).unwrap();

    // send length of serialization to all processes
    let mut serialized_lengths = vec![0i32; world.size() as usize];
    world.all_gather_into(&(serialized.len() as i32), &mut serialized_lengths);

    if world.rank() == 0 as i32 {
        println!("Serialized lengths: {:?}", serialized_lengths);
    }

    // root gathers all serialized trees
    let total_serialized_length = serialized_lengths.iter().sum::<i32>() as usize;
    let mut all_trees_buf = vec![0u8; total_serialized_length];
    let offsets: Vec<i32> = serialized_lengths
        .iter()
        .scan(0, |acc, &x| {
            let tmp = *acc;
            *acc += x;
            Some(tmp)
        })
        .collect();
    let mut partition = PartitionMut::new(&mut all_trees_buf[..], serialized_lengths, &offsets[..]);
    world.all_gather_varcount_into(&serialized, &mut partition);

    // each process deserializes all trees
    let all_trees = offsets
        .iter()
        .enumerate()
        .map(|(i, offset)| {
            if i == world.rank() as usize {
                // just take empty tree here, to skip deserialization of the
                // tree that was created by the process itself.
                // Later, all trees will be merged into the process-local root.
                return root_copy.clone();
            }

            let end_offset = if i == world.size() as usize - 1 {
                total_serialized_length
            } else {
                offsets[i + 1] as usize
            };
            bitcode::deserialize::<TreeNode>(&all_trees_buf[*offset as usize..end_offset]).unwrap()
        })
        .collect::<Vec<TreeNode>>();

    if world.rank() == 0_i32 {
        current_time = mpi::time();
        println!(
            "All trees shared and parsed! time since step started: {} sec",
            current_time - start_time
        );
        start_time = current_time;
    }

    // merge all parsed trees into the local root tree, consuming the parsed trees
    for tree in all_trees {
        root.merge(tree);
    }

    if world.rank() == 0_i32 {
        current_time = mpi::time();
        println!(
            "Trees merged! (Height: {}) time since step started: {} sec",
            root.height(),
            current_time - start_time
        );
        start_time = current_time;
    }

    // calculate forces, velocity and positions for given range
    for b in local_bodies {
        if b.mass == 0f64 {
            continue;
        }

        let f = root.calculate_force(b, theta);
        b.velocity = calc_velocity(&b.velocity, &f, b.mass, timestep);
        b.position = calc_position(&b.velocity, &b.position, timestep);
    }

    if world.rank() == 0_i32 {
        current_time = mpi::time();
        println!(
            "Forces calculated! time since step started: {} sec",
            current_time - start_time
        );
    }
}

fn main() {
    // parse hyperparameteres; shared between all processes without sending them actively
    let args = Args::parse();

    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let root_proc = world.process_at_rank(ROOT_RANK as i32);
    let n_proc = world.size() as usize;
    let rank = world.rank() as usize;

    let start_time = mpi::time();

    // we add zero weight bodies at the end
    // so that all processes get the same amount of bodies
    let bodies_per_proc = (args.n_bodies as f64 / n_proc as f64).ceil() as usize;
    let filled_n = bodies_per_proc * n_proc;
    let extra_n = filled_n - args.n_bodies;

    let mut all_bodies = vec![Body::default(); filled_n];

    // root creates input
    if rank == ROOT_RANK {
        let mut masses = generate_random_bounded(args.n_bodies, 0f64, args.mass_max);
        masses.extend(repeat(0f64).take(extra_n));

        let mut all_positions =
            generate_random_bounded(args.n_bodies * 2, -args.pos_max, args.pos_max);
        all_positions.extend(repeat(0f64).take(extra_n * 2));

        let mut init_velocities =
            generate_random_bounded(args.n_bodies * 2, -args.velocity_max, args.velocity_max);
        init_velocities.extend(repeat(0f64).take(extra_n * 2));

        for i in 0..filled_n {
            let b = &mut all_bodies[i];
            b.id = i;
            b.mass = masses[i];
            b.position = all_positions[i * 2..(i + 1) * 2].try_into().unwrap();
            b.velocity = init_velocities[i * 2..(i + 1) * 2].try_into().unwrap();
        }
    }

    // share all bodies with other processes
    root_proc.broadcast_into(&mut all_bodies);

    let local_range = rank * bodies_per_proc..(rank + 1) * bodies_per_proc;
    let mut local_bodies: Vec<Body> = all_bodies[local_range.clone()].into();

    for _step in 0..args.n_steps {
        // initial tree root
        let mut tree = TreeNode::default();
        tree.center = calc_center(&all_bodies);
        tree.size = get_size(
            &all_bodies
                .iter()
                .map(|b| b.position)
                .collect::<Vec<[f64; 2]>>(),
        );

        barnes_hut(
            &world,
            args.step_time,
            args.theta,
            &mut local_bodies,
            &mut tree,
        );

        // all gather to share updated bodies
        world.all_gather_into(&local_bodies, &mut all_bodies);
    }

    if rank == ROOT_RANK {
        println!("It took {} seconds!", mpi::time() - start_time);
    }
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
