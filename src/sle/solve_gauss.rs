use std::{
    fmt::Debug,
    ops::{DivAssign, SubAssign},
};

use num_traits::{Float, FloatErrorKind, Inv};

pub fn solve_gauss<T>(
    a_mat: &Vec<Vec<T>>,
    b: &Vec<T>,
    eps: &Option<T>,
) -> Result<Vec<T>, FloatErrorKind>
where
    T: Float + Debug + SubAssign + DivAssign + Inv,
{
    assert!(a_mat.len() > 1);
    assert_eq!(
        a_mat.len(),
        a_mat
            .get(0)
            .expect("Unreachable in get(0) after checking size")
            .len()
    );
    let actual_eps = eps.or(Some(T::min_positive_value())).expect("Unreachable");
    let mut eliminated_mat: Vec<Vec<T>> = a_mat.iter().map(|inner_vec| inner_vec.clone()).collect();
    let mut eliminated_b = b.clone();

    forward_elimination(&mut eliminated_mat, &mut eliminated_b, &actual_eps);
    backward_elimination(&mut eliminated_mat, &mut eliminated_b, &actual_eps).map(|()| eliminated_b)
}

fn find_pivot<T>(a_mat: &Vec<Vec<T>>, column: usize) -> usize
where
    T: Float + Debug,
{
    a_mat
        .iter()
        .map(|x| x[column].abs())
        .enumerate()
        .max_by(|(_, x), (_, y)| x.partial_cmp(y).expect("Compared NaN with actual value"))
        .map(|(idx, _)| idx)
        .expect("Should be a non-empty row")
}

fn forward_elimination<T>(a_mat: &mut Vec<Vec<T>>, b: &mut Vec<T>, eps: &T)
where
    T: Float + Debug + SubAssign,
{
    let n = a_mat.len();
    let mut pivot_row;
    let mut factor;
    let mut diff;

    for i in 0..n {
        pivot_row = find_pivot(&a_mat, i);
        a_mat.swap(i, pivot_row);
        a_mat.swap(i, pivot_row);
        for j in (i + 1)..n {
            if a_mat[i][i].abs().gt(eps) {
                factor = a_mat[j][i] / a_mat[i][i];
                for k in i..n {
                    diff = factor * a_mat[i][k];
                    a_mat[j][k] -= diff;
                }
                diff = factor * b[i];
                b[j] -= diff;
            }
        }
    }
}

fn backward_elimination<T>(
    a_mat: &mut Vec<Vec<T>>,
    b: &mut Vec<T>,
    eps: &T,
) -> Result<(), FloatErrorKind>
where
    T: Float + Debug + DivAssign + SubAssign + Inv,
{
    let n = a_mat.len();
    let mut factor;
    for i in (0..n).rev() {
        if a_mat[i][i].abs().lt(eps) {
            return Err(FloatErrorKind::Invalid);
        }
        b[i] /= a_mat[i][i];
        a_mat[i][i].set_one();
        for j in (0..i).rev() {
            factor = b[i] * a_mat[j][i];
            b[j] -= factor;
            a_mat[j][i].set_zero();
        }
    }
    Ok(())
}
