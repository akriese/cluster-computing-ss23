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
        let result_matrix = root_workflow(&world);

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

        // print_matrix(&result_matrix);
        println!("It took {} seconds to finish!", mpi::time() - start_time);
    } else {
        // behavior for all other processes than root ("workers")
        loop {
            let (msg, status): (Vec<u8>, Status) = world.any_process().receive_vec();

            match status.tag() {
                TASK_TAG => handle_task(msg, &world),
                EXIT_TAG => break,
                _ => (),
            }
        }
    }
}

/// The overall work of the root node. Read input, create tasks, distribute them and
/// collect the subresults.
/// If there is no other process in the system, everything is done locally.
///
/// * `world`: MPI communicator to send request over.
fn root_workflow(world: &mpi::topology::SimpleCommunicator) -> Matrix {
    let (a, b, stride) = read_input();

    if world.size() == 1 {
        return calculate_whole_multiplication(&a, &b, None);
    }

    let (m, n) = (a.len(), b[0].len());
    let tasks = create_tasks(&a, &b, stride);
    distribute_and_collect(&tasks, world, m, n, stride)
}

/// Performs the matrix multiplication in one go.
///
/// * `a`: First matrix.
/// * `b`: Second matrix.
fn calculate_whole_multiplication(a: &Matrix, b: &Matrix, transpose_b: Option<bool>) -> Matrix {
    let b_transposed: &Matrix;
    let mut _bt: Matrix;
    if transpose_b.unwrap_or(true) {
        _bt = matrix_transpose(b);
        b_transposed = &_bt;
    } else {
        b_transposed = b;
    };

    let mut result = vec![vec![0.0; b_transposed.len()]; a.len()];

    for (i, row) in a.iter().enumerate() {
        for (j, column) in b_transposed.iter().enumerate() {
            result[i][j] = multiply_row_by_column(row, column);
        }
    }

    result
}

/// Distribute all tasks asynchronously and simultaneosly wait for the answers to come
/// back from the worker processes. Under the hood, the operations MPI_Isend and MPI_Irecv
/// are used to enable the quickest processing.
///
/// * `tasks`: List of tasks serialized to json strings.
/// * `world`: rsmpi communicator to send requests over.
/// * `m`: Number of rows in the result matrix.
/// * `n`: Number of columns in the result matrix.
fn distribute_and_collect(
    tasks: &Vec<String>,
    world: &mpi::topology::SimpleCommunicator,
    m: usize,
    n: usize,
    stride: usize,
) -> Matrix {
    let n_proc = world.size();

    // we need buffers for the immediate_receive to store the message in.
    // the json encoding takes up to 25 bytes for each number. We are generous with the
    // extra 100 bytes for the encoding.
    let recv_buffer_size = 100 + 25 * stride * stride;
    let mut result_collection: Vec<Vec<u8>> = vec![vec![0; recv_buffer_size]; tasks.len()];
    let mut result_matrix: Matrix = vec![vec![0.; n]; m];

    mpi::request::multiple_scope(2 * tasks.len() as usize, |scope, coll| {
        // start all send subtask ops asynchronously
        for (i, task) in tasks.iter().enumerate() {
            let sreq = world
                .process_at_rank(i as i32 % (n_proc - 1) + 1)
                .immediate_send_with_tag(scope, &task.as_bytes()[..], TASK_TAG);
            coll.add(sreq);
        }

        // start all receive subresult ops asynchronously
        for (i, recv_buf) in result_collection.iter_mut().enumerate() {
            let rreq = world
                .process_at_rank(i as i32 % (n_proc - 1) + 1)
                .immediate_receive_into_with_tag(scope, &mut recv_buf[..], RESULT_TAG);

            coll.add(rreq);
        }

        // wait for all requests and process the results from receive operations
        while coll.incomplete() > 0 {
            let (r_id, _status, msg) = coll.wait_any().unwrap();

            // skip the send requests
            if r_id < tasks.len() {
                continue;
            }

            // search for the last actual position of the message
            // inside the fixed length buffer
            let end_pos = msg.iter().position(|c| *c == 0 as u8).unwrap();
            let result: Subresult =
                serde_json::from_str(str::from_utf8(&msg[..end_pos]).unwrap()).unwrap();

            let (m, n) = (result.result.len(), result.result[0].len());
            for i in 0..m {
                for j in 0..n {
                    result_matrix[result.index.0 + i][result.index.1 + j] = result.result[i][j];
                }
            }
        }
    });

    result_matrix
}

/// Handle the subtask in the given message and send back the result to the root.
///
/// * `msg`: Incoming message containing the serialized task.
/// * `world`: MPI communicator object.
fn handle_task(msg: Vec<u8>, world: &mpi::topology::SimpleCommunicator) {
    let task: Subtask = serde_json::from_str(str::from_utf8(msg.as_slice()).unwrap()).unwrap();

    // calculate the value of the result matrix at task.index
    let result = calculate_whole_multiplication(&task.rows, &task.columns, Some(false));

    let send_result = Subresult {
        index: task.index,
        result,
    };

    let serialized = serde_json::to_string(&send_result).unwrap();

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
/// The arguments to this program are expected to be:
/// 1. the file path to the json file containing the input matrices.
/// 2. the stride (integer) used to bundle subtasks.
///
/// If no file path is provided, the returned matrices are small default.
fn read_input() -> (Matrix, Matrix, usize) {
    let args: Vec<String> = env::args().collect();
    println!("{:?}", args);
    let (a, b) = if args.len() > 1 {
        let input_file = File::open(&args[1]).unwrap();
        let buf_reader = BufReader::new(input_file);
        let input_matrices: InputMatrices = serde_json::from_reader(buf_reader).unwrap();
        (input_matrices.a, input_matrices.b)
    } else {
        (
            vec![vec![1., 2.], vec![4., 5.], vec![7., 8.], vec![9., 0.]],
            vec![vec![1., 2., 3., 4., 9.], vec![6., 7., 8., 9., 0.]],
        )
    };

    let stride: usize = if args.len() == 3 {
        args[2].parse::<usize>().unwrap()
    } else {
        4
    };

    (a, b, stride)
}

/// Performs a pairwise multiplication for two arrays and sums the results.
///
/// * `row`: First array.
/// * `column`: Second array.
fn multiply_row_by_column(row: &Row, column: &Column) -> NumberType {
    assert_eq!(row.len(), column.len());
    row.iter()
        .zip(column.iter())
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

/// Creates tasks of the matrix multiplication.
///
/// Currently, one task is defined by the operations necessary for one position in the
/// resulting matrix.
///
/// Returns the number of created jobs.
///
/// * `a`: First matrix.
/// * `b`: Second matrix.
/// * `stride`: Size of the 2D submatrix to send per task.
fn create_tasks(a: &Matrix, b: &Matrix, stride: usize) -> Vec<String> {
    // dimensions of a: m x p
    let (m, _p) = (a.len(), a[0].len());

    // dimensions of b: p x n
    let (_p, n) = (b.len(), b[0].len());

    println!("A: {}x{}, B: {}x{}, result: {}x{}", m, _p, _p, n, m, n);

    // transpose b for an easier access to the rows
    let b_transposed = matrix_transpose(&b);

    // dimensions of the resulting matrix c: m x n
    // iterate over every combination of rows of 'a' with columns of 'b'
    let mut tasks = vec![];
    for i in (0..m).step_by(stride) {
        for j in (0..n).step_by(stride) {
            let msg = Subtask {
                index: (i, j),
                rows: a
                    .into_iter()
                    .cloned()
                    .skip(i)
                    .take(stride)
                    .collect::<Matrix>(),
                columns: (&b_transposed)
                    .into_iter()
                    .cloned()
                    .skip(j)
                    .take(stride)
                    .collect::<Matrix>(),
            };
            tasks.push(serde_json::to_string(&msg).unwrap());
        }
    }

    tasks
}
