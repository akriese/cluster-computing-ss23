use std::{str, vec};

use mpi::point_to_point::Status;
use mpi::request::WaitGuard;
use mpi::traits::*;
use serde::{Deserialize, Serialize};

mod matrix;
mod root;
mod worker;

type NumberType = f64;
type Row = Vec<NumberType>;
type Column = Vec<NumberType>;
type Matrix = Vec<Row>;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Subresult {
    index: (usize, usize),
    result: Matrix,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Subtask {
    index: (usize, usize),
    rows: Matrix,
    columns: Matrix,
}

#[derive(Deserialize, Debug, Clone)]
struct InputMatrices {
    a: Matrix,
    b: Matrix,
}

const TASK_TAG: i32 = 1;
const RESULT_TAG: i32 = 2;
const EXIT_TAG: i32 = 3;

const ROOT_RANK: i32 = 0;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let start_time = mpi::time();

    if world.rank() == ROOT_RANK {
        let result_matrix = root::root_workflow(&world);

        // the workers expect a vec, lets give them an empty one with the EXIT_TAG
        let dummy: Vec<i32> = vec![];

        // signal all nodes other than root to terminate
        for i in 1..world.size() {
            mpi::request::scope(|scope| {
                let _sreq = WaitGuard::from(
                    world
                        .process_at_rank(i)
                        .immediate_send_with_tag(scope, &dummy, EXIT_TAG),
                );
            });
        }

        matrix::print_matrix(&result_matrix);
        println!("It took {} seconds to finish!", mpi::time() - start_time);
    } else {
        // behavior for all other processes than root ("workers")
        loop {
            let (msg, status): (Vec<u8>, Status) = world.any_process().receive_vec();

            match status.tag() {
                TASK_TAG => worker::handle_task(msg, &world),
                EXIT_TAG => break,
                _ => (),
            }
        }
    }
}
