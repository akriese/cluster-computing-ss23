use mpi::traits::*;

use crate::matrix;

use super::Matrix;
use super::{RESULT_TAG, TASK_TAG};
use std::env;

/// The overall work of the root node. Read input, create tasks, distribute them and
/// collect the subresults.
/// If there is no other process in the system, everything is done locally.
///
/// * `world`: MPI communicator to send request over.
pub(crate) fn root_workflow(world: &mpi::topology::SimpleCommunicator) -> Matrix {
    let (a, b) = read_input();

    if world.size() == 1 {
        return matrix::multiplication(&a, &b, None);
    }

    let (m, n) = (a.len(), b[0].len());
    distribute_and_collect(&a, world, m, n)
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
    a: &Matrix,
    world: &mpi::topology::SimpleCommunicator,
    m: usize,
    n: usize,
) -> Matrix {
    let n_proc = world.size();

    let mut result_buffer: Matrix = vec![vec![0f64; n]; m];
    let mut result_matrix: Matrix = vec![vec![0f64; n]; m];

    mpi::request::multiple_scope(2 * m as usize, |scope, coll| {
        let mut jobs = vec![vec![]; world.size() as usize];

        // start all send subtask ops asynchronously
        for (i, task) in a.iter().enumerate() {
            let proc = i as i32 % (n_proc - 1) + 1;
            let sreq =
                world
                    .process_at_rank(proc)
                    .immediate_send_with_tag(scope, &task[..], TASK_TAG);
            coll.add(sreq);

            jobs[proc as usize].push(i);
        }

        // start all receive subresult ops asynchronously
        for (i, recv_buf) in result_buffer.iter_mut().enumerate() {
            let rreq = world
                .process_at_rank(i as i32 % (n_proc - 1) + 1)
                .immediate_receive_into_with_tag(scope, &mut recv_buf[..], RESULT_TAG);

            coll.add(rreq);
        }

        // wait for all requests and process the results from receive operations
        while coll.incomplete() > 0 {
            let (r_id, _status, msg) = coll.wait_any().unwrap();

            // skip the send requests
            if r_id < m {
                continue;
            }

            let source_rank = _status.source_rank();
            let position = jobs[source_rank as usize].pop().unwrap();

            result_matrix[position] = msg.to_vec();
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
pub(crate) fn read_input() -> (Matrix, Matrix) {
    let args: Vec<String> = env::args().collect();
    let (m, p, n) = (
        args[1].parse::<usize>().unwrap(),
        args[2].parse::<usize>().unwrap(),
        args[3].parse::<usize>().unwrap(),
    );

    let a = matrix::generate_2d(m, p);
    let b = matrix::generate_2d(p, n);

    (a, b)
}
