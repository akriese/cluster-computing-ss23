use std::iter::repeat;

use clap::Parser;
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

    #[arg(short = 't', default_value_t = 0.5)]
    theta: f64,
}

#[derive(Clone, Debug, Equivalence, Default)]
struct Body {
    id: usize,
    mass: f64,
    position: [f64; 2],
    velocity: [f64; 2],
}

#[derive(Clone, Default, Debug)]
struct TreeNode {
    center: [f64; 2],
    size: f64,
    mass: f64,
    mass_center: [f64; 2],
    children: Vec<TreeNode>,
    body: Option<Body>,
}

impl TreeNode {
    fn split(&mut self) {
        let center_offset = self.size / 4 as f64;
        let mut dummy = TreeNode::default();
        dummy.size = self.size / 2 as f64;
        dummy.center = [
            self.center[0] + center_offset,
            self.center[1] + center_offset,
        ];
        self.children.push(dummy.clone());
        dummy.center = [
            self.center[0] - center_offset,
            self.center[1] + center_offset,
        ];
        self.children.push(dummy.clone());
        dummy.center = [
            self.center[0] - center_offset,
            self.center[1] - center_offset,
        ];
        self.children.push(dummy.clone());
        dummy.center = [
            self.center[0] + center_offset,
            self.center[1] - center_offset,
        ];
        self.children.push(dummy);
    }

    fn push_to_child(&mut self, body: &Body) {
        if self.children.len() == 0 {
            self.split();
        }

        if body.position[0] > self.center[0] {
            if body.position[1] > self.center[1] {
                self.children[0].insert(&body);
            } else {
                self.children[3].insert(&body);
            }
        } else {
            if body.position[1] > self.center[1] {
                self.children[1].insert(&body);
            } else {
                self.children[2].insert(&body);
            }
        }
    }

    fn insert(&mut self, body: &Body) {
        if self.children.is_empty() && self.body.is_none() {
            self.body = Some(body.clone());
        } else {
            if let Some(b) = &self.body {
                self.push_to_child(&b.clone());
                self.body = None;
            }

            self.push_to_child(body);
        }

        self.mass += body.mass;
        self.mass_center[0] = (self.mass_center[0] * (self.mass - body.mass)
            + body.position[0] * body.mass)
            / self.mass;
        self.mass_center[1] = (self.mass_center[1] * (self.mass - body.mass)
            + body.position[1] * body.mass)
            / self.mass;
    }

    fn calculate_force(&mut self, body: &Body, theta: f64) -> [f64; 2] {
        let displacement = [
            self.mass_center[0] - body.position[0],
            self.mass_center[1] - body.position[1],
        ];
        let distance =
            (displacement[0] * displacement[0] + displacement[1] * displacement[1]).sqrt();

        if distance == 0.0 {
            return [0f64; 2];
        }

        if let Some(b) = &self.body {
            let f = G * b.mass * body.mass / (distance * distance);
            return [
                f + displacement[0] / distance,
                f + displacement[1] / distance,
            ];
        } else if self.children.len() > 0 {
            if self.size / distance < theta {
                let f = G * self.mass * body.mass / (distance * distance);
                return [
                    f + displacement[0] / distance,
                    f + displacement[1] / distance,
                ];
            } else {
                let mut summed_force = [f64::default(); 2];
                for child in self.children.iter_mut() {
                    let f = child.calculate_force(body, theta);
                    summed_force[0] += f[0];
                    summed_force[1] += f[1];
                }

                return summed_force;
            }
        } else {
            // empty quadrant
            return [0f64; 2];
        }
    }
}

fn calc_center(bodies: &[Body]) -> [f64; 2] {
    let x = bodies.iter().map(|b| b.position[0]).sum::<f64>() / bodies.len() as f64;
    let y = bodies.iter().map(|b| b.position[1]).sum::<f64>() / bodies.len() as f64;

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

fn barnes_hut(bodies: &Vec<Body>, timestep: f64, theta: f64, local_bodies: &mut Vec<Body>) {
    let mut start_time = mpi::time();
    let mut current_time;

    // build the tree
    let mut tree = TreeNode::default();
    tree.center = calc_center(&bodies);
    tree.size = get_size(&bodies.iter().map(|b| b.position).collect::<Vec<[f64; 2]>>());

    for body in bodies {
        if body.mass > 0f64 {
            tree.insert(&body);
        }
    }

    current_time = mpi::time();
    println!("Tree built! took {} sec", current_time - start_time);
    start_time = current_time;

    // calculate forces, velocity and positions for given range
    for b in local_bodies {
        if b.mass == 0f64 {
            continue;
        }

        let f = tree.calculate_force(b, theta);
        b.velocity = calc_velocity(&b.velocity, &f, b.mass, timestep);
        b.position = calc_position(&b.velocity, &b.position, timestep);
    }

    current_time = mpi::time();
    println!("Forces calculated! took {} sec", current_time - start_time);
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

    for step in 0..args.n_steps {
        barnes_hut(&all_bodies, args.step_time, args.theta, &mut local_bodies);

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
