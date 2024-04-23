use std::{str, vec};

use mpi::point_to_point::Status;
use mpi::request::WaitGuard;
use mpi::traits::*;
use serde::{Deserialize, Serialize};

type Row = Vec<i32>;
type Column = Vec<i32>;
type Matrix = Vec<Row>;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Subresult {
    index: (usize, usize),
    result: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Subtask {
    index: (usize, usize),
    row: Row,
    column: Column,
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let root_rank = 0;

    let a = vec![vec![1, 2], vec![4, 5], vec![7, 8], vec![9, 0]];
    let b = vec![vec![1, 2, 3, 4], vec![6, 7, 8, 9]];

    if world.rank() == root_rank {
        distribute_subtasks(a, b, &world);
    }

    loop {
        let (msg, status): (Vec<u8>, Status) = world.any_process().receive_vec();
        let task: Subtask = serde_json::from_str(str::from_utf8(msg.as_slice()).unwrap()).unwrap();

        println!(
            "Process {} got task {:?}.\nStatus is: {:?}",
            rank, task, status
        );

        // calculate the value of the result matrix at task.index
        let result = multiply_row_by_column(task.row, task.column);

        let send_result = Subresult {
            index: task.index,
            result,
        };

        let serialized = serde_json::to_string(&send_result).unwrap();

        // send back the result to the root process
        mpi::request::scope(|scope| {
            let _sreq = world
                .process_at_rank(root_rank)
                .immediate_send(scope, &serialized.as_bytes()[..]);
        });
    }
}

/// Performs a pairwise multiplication for two arrays and sums the results.
///
/// * `row`: First array.
/// * `column`: Second array.
fn multiply_row_by_column(row: Vec<i32>, column: Vec<i32>) -> i32 {
    assert_eq!(row.len(), column.len());
    row.into_iter()
        .zip(column.into_iter())
        .fold(0, |sum, (a, b)| sum + a * b)
}

/// Transposes a 2D matrix.
///
/// * `a`: The [Matrix] to transpose.
fn matrix_transpose(a: &Matrix) -> Matrix {
    assert!(a.len() > 0);
    let (m, n) = (a.len(), a[0].len());

    let mut result: Matrix = vec![vec![0; m]; n];
    for i in 0..n {
        for j in 0..m {
            result[i][j] = a[j][i];
        }
    }

    return result;
}

/// Distributes tasks of the matrix multiplication.
///
/// Currently, one task is defined by the operations necessary for one position in the
/// resulting matrix.
///
/// * `a`: First matrix.
/// * `b`: Second matrix.
/// * `world`: The MPI world object that the program is being run in.
fn distribute_subtasks(a: Matrix, b: Matrix, world: &mpi::topology::SimpleCommunicator) {
    let size = world.size();

    // dimensions of a: m x p
    let (m, _p) = (a.len(), a[0].len());

    // dimensions of b: p x n
    let (_p, n) = (b.len(), b[0].len());

    // transpose b for an easier access to the rows
    let b_translated = matrix_transpose(&b);

    // dimensions of the resulting matrix c: m x n
    // iterate over every combination of rows of 'a' with columns of 'b'
    let mut c = 1;
    for i in 0..m {
        for j in 0..n {
            let msg = Subtask {
                index: (i, j),
                row: a[i].clone(),
                column: b_translated[j].clone(),
            };
            let serialized = serde_json::to_string(&msg).unwrap();

            // send the serialized message
            mpi::request::scope(|scope| {
                let _sreq = WaitGuard::from(
                    world
                        .process_at_rank(c % size)
                        .immediate_send(scope, &serialized.as_bytes()[..]),
                );
            });

            c += 1;
        }
    }
}
