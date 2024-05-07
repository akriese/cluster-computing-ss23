use super::{Subresult, Subtask};
use super::{RESULT_TAG, ROOT_RANK};
use crate::matrix;
use mpi::request::WaitGuard;
use mpi::traits::*;
use std::str;

/// Handle the subtask in the given message and send back the result to the root.
///
/// * `msg`: Incoming message containing the serialized task.
/// * `world`: MPI communicator object.
pub(crate) fn handle_task(msg: Vec<u8>, world: &mpi::topology::SimpleCommunicator) {
    let task: Subtask = serde_json::from_str(str::from_utf8(msg.as_slice()).unwrap()).unwrap();

    // calculate the value of the result matrix at task.index
    let result = matrix::multiplication(&task.rows, &task.columns, Some(false));

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
