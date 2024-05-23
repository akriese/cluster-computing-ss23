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
    // velocity: [f64; 2],
}

struct TreeNode {
    center: [f64; 2],
    mass: f64,
    force: [f64; 2],
    children: Vec<TreeNode>,
    body: Option<Body>,
}

impl TreeNode {
    fn insert(&mut self, body: &Body) {
        match &self.body {
            Some(b) => {
                self.split();
                self.children.push(TreeNode { body: b });
                self.body = None;
            }
            None => {
                self.body = Some(body.clone());
            }
        }
    }

    fn split(&mut self) {}
}

fn build_tree(bodies: &[Body]) -> TreeNode {
    let center = calc_center(bodies);
    let mut root_node = TreeNode {
        center,
        mass: 0f64,
        force: [0f64; 2],
        children: vec![],
        body: None,
    };

    for body in bodies.iter() {
        root_node.insert(body);
    }

    root_node
}

fn calc_center(bodies: &[Body]) -> [f64; 2] {
    [0.0, 0.0]
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
    let filled_n = ((args.n_bodies as f64 / n_proc as f64).ceil() as usize) * n_proc;

    let mut masses: Vec<f64> = vec![0f64; filled_n];
    let mut all_positions: Vec<f64> = vec![0f64; filled_n * 2];

    // root creates input
    if rank == ROOT_RANK {
        masses = generate_random_bounded(filled_n, 0f64, args.mass_max);

        // reset positions of phantom filled bodies
        masses[args.n_bodies..filled_n].fill(0f64);
        all_positions = generate_random_bounded(filled_n * 2, -args.pos_max, args.pos_max);
    }

    if rank == ROOT_RANK && args.print {
        println!("Masses: {:?}", masses);
    }

    // root sends masses
    root_proc.broadcast_into(&mut masses);

    let bodies_per_proc = filled_n / n_proc as usize;

    // root sends initial coordinates to everyone
    root_proc.broadcast_into(&mut all_positions);

    if rank == ROOT_RANK && args.print {
        println!("{:?}", all_positions);
    }

    // root sends initial velocity to respective ranks
    let mut local_velocities = vec![0f64; bodies_per_proc * 2];
    if rank == ROOT_RANK {
        let mut init_velocities =
            generate_random_bounded(filled_n * 2, -args.velocity_max, args.velocity_max);

        // reset velocities of phantom filled bodies
        init_velocities[args.n_bodies..filled_n].fill(0f64);
        root_proc.scatter_into_root(&init_velocities, &mut local_velocities);
    } else {
        root_proc.scatter_into(&mut local_velocities);
    }

    let mut local_positions =
        all_positions[rank * bodies_per_proc * 2..(rank + 1) * bodies_per_proc * 2].to_vec();

    for _t in 0..args.n_steps {
        // calculate their velocity and positions
        (local_positions, local_velocities) = calculate_next_step(
            &local_velocities,
            &local_positions,
            rank,
            &masses,
            &all_positions,
            args.step_time,
        );

        // send new positions with MPI_Allgather
        world.all_gather_into(&local_positions, &mut all_positions);
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
