use std::vec;

use nume_rs::{ops::basic::{MatMul, VecElementwiseArithmetic}, sle::solve_gauss::solve_gauss};


fn main() {
    let a_mat = vec![
        vec![4.0f32, 1.0, 3.59],
        vec![4.10, -1.24, 3.46],
        vec![-1.2, 5.12, 1.31]
    ];
    let b = vec![0.51, 1.64, 7.62];
    let x = solve_gauss(&a_mat, &b, &None).unwrap();
    let mut out = a_mat.matmul(&x).unwrap().elementwise_sub(&b).unwrap();
    
    println!("Vecs: b = {:?} A*x = {:?}", b, a_mat.matmul(&x).unwrap());
    println!("Squared error: {:?}", out.iter().map(|x| x*x).reduce(|a, b| a + b).unwrap() / (b.len() as f32));
}
