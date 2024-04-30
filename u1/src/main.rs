use std::{env, fs::File, io::BufReader, str, vec};

use mpi::point_to_point::Status;
use mpi::request::WaitGuard;
use mpi::traits::*;
use serde::{Deserialize, Serialize};

type NumberType = f64;
type Row = Vec<NumberType>;
type Column = Vec<NumberType>;
type Matrix = Vec<Row>;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Subresult {
    index: (usize, usize),
    result: NumberType,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Subtask {
    index: (usize, usize),
    row: Row,
    column: Column,
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

    let (mut m, mut n) = (0, 0);
    if world.rank() == ROOT_RANK {
        let (a, b) = read_input();
        (m, n) = (a.len(), b[0].len());
        distribute_subtasks(&a, &b, &world);
    }

    let mut result_matrix: Matrix = vec![vec![0.; n]; m];
    let mut count_received = 0;

    loop {
        let (msg, status): (Vec<u8>, Status) = world.any_process().receive_vec();

        match status.tag() {
            RESULT_TAG => {
                println!("received result from rank {}", status.source_rank());
                let result: Subresult =
                    serde_json::from_str(str::from_utf8(msg.as_slice()).unwrap()).unwrap();

                result_matrix[result.index.0][result.index.1] = result.result;
                count_received += 1;

                if count_received == m * n {
                    println!("It's all over!");
                    break;
                }

                continue;
            }
            TASK_TAG => handle_task(msg, status, &world),
            EXIT_TAG => break,
            _ => (),
        }
    }

    // signal all nodes other than root to terminate
    if world.rank() == ROOT_RANK {
        print_matrix(&result_matrix);
        println!("It took {} seconds to finish!", mpi::time() - start_time);

        let dummy: Vec<i32> = vec![];
        for i in 1..world.size() {
            mpi::request::scope(|scope| {
                let _sreq = WaitGuard::from(
                    world
                        .process_at_rank(i)
                        .immediate_send_with_tag(scope, &dummy, EXIT_TAG),
                );
            });
        }
    }
}

/// Handle the subtask in the given message and send back the result to the root.
///
/// * `msg`: Incoming message containing the serialized task.
/// * `status`: Status of the incoming message.
/// * `world`: MPI communicator object.
fn handle_task(msg: Vec<u8>, status: Status, world: &mpi::topology::SimpleCommunicator) {
    let task: Subtask = serde_json::from_str(str::from_utf8(msg.as_slice()).unwrap()).unwrap();

    println!(
        "Process {} got task {:?}.\nStatus is: {:?}",
        world.rank(),
        &task,
        status
    );

    // calculate the value of the result matrix at task.index
    let result = multiply_row_by_column(task.row, task.column);

    let send_result = Subresult {
        index: task.index,
        result,
    };

    let serialized = serde_json::to_string(&send_result).unwrap();
    println!("{}", &serialized);

    // send back the result to the root process
    mpi::request::scope(|scope| {
        let _sreq = WaitGuard::from(world.process_at_rank(ROOT_RANK).immediate_send_with_tag(
            scope,
            &serialized.as_bytes()[..],
            RESULT_TAG,
        ));
    });
}

/// Prints a 2D matrix in the classical representation (rows are stacked vertically).
///
/// * `a`: The matrix to be printed.
fn print_matrix(a: &Matrix) {
    for row in a {
        println!("{:?}", row);
    }
}

/// Reads the input json file if provided by a command line argument.
///
/// If no file path is provided, the returned matrices are small default.
fn read_input() -> (Matrix, Matrix) {
    let args: Vec<String> = env::args().collect();
    println!("{:?}", args);
    if args.len() > 1 {
        let input_file = File::open(&args[1]).unwrap();
        let buf_reader = BufReader::new(input_file);
        let input_matrices: InputMatrices = serde_json::from_reader(buf_reader).unwrap();
        (input_matrices.a, input_matrices.b)
    } else {
        (
            vec![vec![1., 2.], vec![4., 5.], vec![7., 8.], vec![9., 0.]],
            vec![vec![1., 2., 3., 4., 9.], vec![6., 7., 8., 9., 0.]],
        )
    }
}

/// Performs a pairwise multiplication for two arrays and sums the results.
///
/// * `row`: First array.
/// * `column`: Second array.
fn multiply_row_by_column(row: Row, column: Column) -> NumberType {
    assert_eq!(row.len(), column.len());
    row.into_iter()
        .zip(column.into_iter())
        .fold(0., |sum, (a, b)| sum + a * b)
}

/// Transposes a 2D matrix.
///
/// * `a`: The [Matrix] to transpose.
fn matrix_transpose(a: &Matrix) -> Matrix {
    assert!(a.len() > 0);
    let (m, n) = (a.len(), a[0].len());

    let mut result: Matrix = vec![vec![0.; m]; n];
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
fn distribute_subtasks(a: &Matrix, b: &Matrix, world: &mpi::topology::SimpleCommunicator) {
    let size = world.size();

    // dimensions of a: m x p
    let (m, _p) = (a.len(), a[0].len());

    // dimensions of b: p x n
    let (_p, n) = (b.len(), b[0].len());

    println!("A: {}x{}, B: {}x{}, result: {}x{}", m, _p, _p, n, m, n);

    // transpose b for an easier access to the rows
    let b_transpose = matrix_transpose(&b);

    // dimensions of the resulting matrix c: m x n
    // iterate over every combination of rows of 'a' with columns of 'b'
    let mut c = 0;
    for i in 0..m {
        for j in 0..n {
            let msg = Subtask {
                index: (i, j),
                row: a[i].clone(),
                column: b_transpose[j].clone(),
            };
            let serialized = serde_json::to_string(&msg).unwrap();

            // send the serialized message
            mpi::request::scope(|scope| {
                let _sreq =
                    WaitGuard::from(world.process_at_rank(c % size).immediate_send_with_tag(
                        scope,
                        &serialized.as_bytes()[..],
                        TASK_TAG,
                    ));
            });

            c += 1;
        }
    }
}
