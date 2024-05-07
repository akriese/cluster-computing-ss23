use crate::matrix;

use super::{InputMatrices, Matrix, Subresult, Subtask};
use super::{RESULT_TAG, TASK_TAG};
use mpi::traits::*;
use std::{env, fs::File, io::BufReader, str};

/// The overall work of the root node. Read input, create tasks, distribute them and
/// collect the subresults.
/// If there is no other process in the system, everything is done locally.
///
/// * `world`: MPI communicator to send request over.
pub(crate) fn root_workflow(world: &mpi::topology::SimpleCommunicator) -> Matrix {
    let (a, b, stride) = read_input();

    if world.size() == 1 {
        return matrix::multiplication(&a, &b, None);
    }

    let (m, n) = (a.len(), b[0].len());
    let tasks = create_tasks(&a, &b, stride);
    distribute_and_collect(&tasks, world, m, n, stride)
}

/// Distribute all tasks asynchronously and simultaneosly wait for the answers to come
/// back from the worker processes. Under the hood, the operations MPI_Isend and MPI_Irecv
/// are used to enable the quickest processing.
///
/// * `tasks`: List of tasks serialized to json strings.
/// * `world`: rsmpi communicator to send requests over.
/// * `m`: Number of rows in the result matrix.
/// * `n`: Number of columns in the result matrix.
pub(crate) fn distribute_and_collect(
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

/// Reads the input json file if provided by a command line argument.
///
/// The arguments to this program are expected to be:
/// 1. the file path to the json file containing the input matrices.
/// 2. the stride (integer) used to bundle subtasks.
///
/// If no file path is provided, the returned matrices are small default.
pub(crate) fn read_input() -> (Matrix, Matrix, usize) {
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
pub(crate) fn create_tasks(a: &Matrix, b: &Matrix, stride: usize) -> Vec<String> {
    // dimensions of a: m x p
    let (m, _p) = (a.len(), a[0].len());

    // dimensions of b: p x n
    let (_p, n) = (b.len(), b[0].len());

    println!("A: {}x{}, B: {}x{}, result: {}x{}", m, _p, _p, n, m, n);

    // transpose b for an easier access to the rows
    let b_transposed = matrix::matrix_transpose(&b);

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
