use clap::Parser;
use itertools::Itertools;
use mpi::traits::*;
use rand::{thread_rng, Rng};

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
}

#[derive(Clone, Debug)]
struct Body {
    id: usize,
    mass: f64,
    position: [f64; 2],
    velocity: [f64; 2],
}

struct TreeNode {
    center: [f64; 2],
    mass: f64,
    force: [f64; 2],
    children: Vec<TreeNode>,
    body: Option<Body>,
}

impl TreeNode {
    fn from_bodies(bodies: &[Body]) -> TreeNode {
        let n = bodies.len();
        match n {
            0 => TreeNode {
                center: [0f64, 0f64],
                mass: 0f64,
                children: vec![],
                body: None,
                force: [0f64, 0f64],
            },
            1 => TreeNode {
                center: bodies[0].position,
                mass: bodies[0].mass,
                children: vec![],
                body: Some(bodies[0].clone()),
                force: [0f64, 0f64],
            },
            _ => {
                let center = calc_center(bodies);
                TreeNode {
                    center,
                    mass: 0f64,
                    force: [0f64; 2],
                    children: vec![
                        TreeNode::from_bodies(
                            &bodies
                                .into_iter()
                                .filter(|b| {
                                    b.position[0] >= center[0] && b.position[1] >= center[1]
                                })
                                .cloned()
                                .collect::<Vec<_>>()[..],
                        ),
                        TreeNode::from_bodies(
                            &bodies
                                .into_iter()
                                .filter(|b| b.position[0] < center[0] && b.position[1] >= center[1])
                                .cloned()
                                .collect::<Vec<_>>(),
                        ),
                        TreeNode::from_bodies(
                            &bodies
                                .into_iter()
                                .filter(|b| b.position[0] < center[0] && b.position[1] < center[1])
                                .cloned()
                                .collect::<Vec<_>>(),
                        ),
                        TreeNode::from_bodies(
                            &bodies
                                .into_iter()
                                .filter(|b| b.position[0] >= center[0] && b.position[1] < center[1])
                                .cloned()
                                .collect::<Vec<_>>(),
                        ),
                    ],
                    body: None,
                }
            }
        }
    }
}

fn calc_center(bodies: &[Body]) -> [f64; 2] {
    let x = bodies.iter().map(|b| b.position[0]).sum::<f64>() / bodies.iter().len() as f64;
    let y = bodies.iter().map(|b| b.position[1]).sum::<f64>() / bodies.iter().len() as f64;

    return [x, y];
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

fn get_center_of_mass(root: &TreeNode) {}

fn barnes_hut(root: &TreeNode) {
    // calculate mass centers from leaves up
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

    let mut masses: Vec<f64> = vec![0f64; args.n_bodies];
    let mut all_positions: Vec<f64> = vec![0f64; args.n_bodies * 2];
    let init_velocities: Vec<f64> = vec![0f64; args.n_bodies * 2];

    let mut bodies = vec![];
    for i in 0..args.n_bodies {
        bodies.push(Body {
            id: i,
            mass: masses[i],
            position: all_positions[i * 2..(i + 1) * 2].try_into().unwrap(),
            velocity: init_velocities[i * 2..(i + 1) * 2].try_into().unwrap(),
        })
    }

    let tree = TreeNode::from_bodies(&bodies);

    barnes_hut(&tree);

    // root creates input
    if rank == ROOT_RANK {
        masses = generate_random_bounded(args.n_bodies, 0f64, args.mass_max);

        all_positions = generate_random_bounded(args.n_bodies * 2, -args.pos_max, args.pos_max);
    }

    if rank == ROOT_RANK && args.print {
        println!("Masses: {:?}", masses);
    }

    // root sends masses
    root_proc.broadcast_into(&mut masses);

    // root sends initial coordinates to everyone
    root_proc.broadcast_into(&mut all_positions);

    if rank == ROOT_RANK && args.print {
        println!("{:?}", all_positions);
    }

    for _t in 0..args.n_steps {
        // calculate their velocity and positions
        // (local_positions, local_velocities) = calculate_next_step(
        //     &local_velocities,
        //     &local_positions,
        //     rank,
        //     &masses,
        //     &all_positions,
        //     args.step_time,
        // );

        // send new positions with MPI_Allgather
        // world.all_gather_into(&local_positions, &mut all_positions);
        world.barrier();

        if rank == ROOT_RANK && args.print {
            println!("{:?}", all_positions);
        }
    }

    if rank == ROOT_RANK {
        println!("It took {} seconds!", mpi::time() - start_time);
    }
}

/// Calculate forces, new velocities and positions for a subset of bodies for one timestep.
///
/// * `local_velocities`: Velocities of the bodies to calculate values for.
/// * `local_positions`: Positions of the bodies to calculate values for.
/// * `local_offset`: Offset of given package of bodies in the complete array of bodies.
/// * `masses`: All masses
/// * `all_positions`: All bodies' positions
/// * `timestep`: Size of the time step in s
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

/// Calculate the new velocity of a body.
///
/// * `old_velocity`: Old velocity
/// * `force`: Current force on the body
/// * `mass`: Body's mass
/// * `timestep`: Step size of the time
fn calc_velocity(old_velocity: &[f64; 2], force: &[f64; 2], mass: f64, timestep: f64) -> [f64; 2] {
    let [v_x, v_y] = old_velocity;
    let [f_x, f_y] = force;
    return [v_x + f_x / mass * timestep, v_y + f_y / mass * timestep];
}

/// Calculate the new position of a body.
///
/// * `velocity`: New velocity
/// * `old_position`: Old position
/// * `timestep`: Time step size
fn calc_position(velocity: &[f64; 2], old_position: &[f64; 2], timestep: f64) -> [f64; 2] {
    let [v_x, v_y] = velocity;
    let [x, y] = old_position;
    return [x + v_x * timestep, y + v_y * timestep];
}

/// Calculate the force on one body. This is the trivial approach leading to a running
/// time of N^2 per step. One could cut that down to N*(N-1)/2 as the handshake lemma can be
/// applied.
///
/// * `self_position`: Position of the body
/// * `self_mass`: Mass of the body
/// * `other_positions`: Positions of the other bodies
/// * `masses`: Masses of the other bodies
fn calc_force(
    self_position: &[f64; 2],
    self_mass: f64,
    other_positions: &Vec<f64>,
    masses: &Vec<f64>,
) -> [f64; 2] {
    let mut summed_force = [0f64; 2];
    let [self_x, self_y] = self_position;
    for (i, (x, y)) in other_positions.iter().tuples().enumerate() {
        // distances per axis
        let (d_x, d_y) = (x - self_x, y - self_y);

        // avoid division by zero (e.g. if the body itself is in the array)
        if d_x == 0f64 && d_y == 0f64 {
            continue;
        }

        // euclidean distance
        let r = (d_x * d_x + d_y * d_y).sqrt();

        // Force after Newton's first law
        let f = G * self_mass * masses[i] / (r * r);

        // add directional force to the collective sum
        summed_force[0] += f + d_x / r;
        summed_force[1] += f + d_y / r;
    }

    return summed_force;
}
