use std::vec;

use nume_rs::sle::solve_gauss::solve_gauss;

fn main() {
    let A = vec![
        vec![4.0f32, 1.0, 3.59],
        vec![4.10, -1.24, 3.46],
        vec![-1.2, 5.12, 1.31]
    ];
    let b = vec![0.51, 1.64, 7.62];
    println!("{:?}", solve_gauss(&A, &b, &None));
}
