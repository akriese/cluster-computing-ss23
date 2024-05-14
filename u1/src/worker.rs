use super::{RESULT_TAG, ROOT_RANK};
use crate::{matrix, Matrix};
use mpi::{
    request::WaitGuard,
    traits::{Communicator, Destination},
};

/// Handle the subtask in the given message and send back the result to the root.
///
/// * `a`: Matrix (usually one row)
/// * `b`: Matrix (usually whole second matrix)
/// * `world`: MPI communicator object.
pub(crate) fn handle_task(a: &Matrix, b: &Matrix, world: &mpi::topology::SimpleCommunicator) {
    // calculate the value of the result matrix at task.index
    let result = matrix::multiplication(a, b, Some(false));

    // send back the result to the root process
    mpi::request::scope(|scope| {
        let _sreq = WaitGuard::from(
            world
                .process_at_rank(ROOT_RANK)
                .immediate_send_with_tag(scope, &result[0], RESULT_TAG),
        );
    });
}
