use super::{Column, Matrix, NumberType, Row};
use rand::{thread_rng, Rng};

/// Prints a 2D matrix in the classical representation (rows are stacked vertically).
///
/// * `a`: The matrix to be printed.
pub(crate) fn print_matrix(a: &Matrix) {
    for _row in a {
        // println!("{:?}", _row);
    }
}

/// Performs a pairwise multiplication for two arrays and sums the results.
///
/// * `row`: First array.
/// * `column`: Second array.
pub(crate) fn multiply_row_by_column(row: &Row, column: &Column) -> NumberType {
    assert_eq!(row.len(), column.len());
    row.iter()
        .zip(column.iter())
        .fold(0., |sum, (a, b)| sum + a * b)
}

/// Performs the matrix multiplication in one go.
///
/// * `a`: First matrix.
/// * `b`: Second matrix.
pub(crate) fn multiplication(a: &Matrix, b: &Matrix, transpose_b: Option<bool>) -> Matrix {
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

/// Transposes a 2D matrix.
///
/// * `a`: The [Matrix] to transpose.
pub(crate) fn matrix_transpose(a: &Matrix) -> Matrix {
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

pub(crate) fn generate_2d(m: usize, n: usize) -> Matrix {
    let mut result = vec![vec![0.0; n]; m];
    for row in result.iter_mut() {
        thread_rng().fill(&mut row[..]);
    }

    result
}
