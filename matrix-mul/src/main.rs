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

const TASK_TAG: i32 = 1;
const RESULT_TAG: i32 = 2;
const EXIT_TAG: i32 = 3;

const ROOT_RANK: i32 = 0;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let start_time = mpi::time();

    let (a, mut b);
    let mut buf = [0usize; 3];
    let mut bbuf: Vec<NumberType>;

    if world.rank() == ROOT_RANK {
        (a, b) = root::read_input();
        let (m, p, n) = (a.len(), b.len(), b[0].len());
        buf = [m, p, n];
    } else {
        b = vec![]; // prevent rust warning about possibly uninitialized array
    }

    // broadcast dimensions
    world.process_at_rank(ROOT_RANK).broadcast_into(&mut buf);
    let [_m, p, n] = buf;

    if world.rank() == ROOT_RANK {
        bbuf = b.into_iter().flatten().collect();
    } else {
        bbuf = vec![0 as NumberType; p * n];
    }

    // distribute matrix b
    world.process_at_rank(ROOT_RANK).broadcast_into(&mut bbuf);

    b = vec![];
    for i in 0..p {
        b.push(bbuf[n * i..n * (i + 1)].to_vec());
    }

    // distribute, calculate and collect
    if world.rank() == ROOT_RANK {
        world.process_at_rank(ROOT_RANK).scatter_into_root(sendbuf, recvbuf)
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
        // transpose the matrix once
        b = matrix::matrix_transpose(&b);

        loop {
            let (msg, status): (Row, Status) = world.any_process().receive_vec();

            let row_as_matrix = vec![msg];

            match status.tag() {
                TASK_TAG => worker::handle_task(&row_as_matrix, &b, &world),
                EXIT_TAG => break,
                _ => (),
            }
        }
    }
}
