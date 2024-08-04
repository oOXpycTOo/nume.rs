use std::{
    ops::{Add, AddAssign, Mul, Sub},
    process::Output,
    vec,
};

use num_traits::{Float, Zero};

#[derive(Debug)]
pub enum MathError {
    DimensionMismatch,
    EmptyInput,
}

pub trait MatMul<Rhs> {
    type Output;

    fn matmul(&self, rhs: &Rhs) -> Result<Self::Output, MathError>;
}

impl<T> MatMul<Vec<T>> for Vec<Vec<T>>
where
    T: Copy + AddAssign + Zero + Mul<Output = T>,
{
    type Output = Vec<T>;

    fn matmul(&self, rhs: &Vec<T>) -> Result<Self::Output, MathError> {
        if self.len() == 0 || rhs.len() == 0 {
            return Err(MathError::EmptyInput);
        }
        if self[0].len() != rhs.len() {
            return Err(MathError::DimensionMismatch);
        }
        let rows = self.len();
        let cols = rhs.len();
        let mut result = vec![T::zero(); rows];
        for i in 0..rows {
            for j in 0..cols {
                result[i] += self[i][j] * rhs[j];
            }
        }
        Ok(result)
    }
}

pub trait ElementwiseOp<T> {
    fn elementwise_op<F>(&self, rhs: &Self, op: F) -> Result<Vec<T>, MathError>
    where
        F: Fn(T, T) -> T;
}

impl<T> ElementwiseOp<T> for Vec<T>
where
    T: Copy,
{
    fn elementwise_op<F>(&self, rhs: &Self, op: F) -> Result<Vec<T>, MathError>
    where
        F: Fn(T, T) -> T,
    {
        if self.len() != rhs.len() {
            return Err(MathError::DimensionMismatch);
        }

        Ok(self
            .iter()
            .zip(rhs.iter())
            .map(|(&a, &b)| op(a, b))
            .collect())
    }
}

pub trait VecElementwiseArithmetic<T>
where
    Self: ElementwiseOp<T>,
    T: Add<Output = T> + Sub<Output = T> + Copy,
{
    fn elementwise_add(&self, rhs: &Self) -> Result<Vec<T>, MathError> {
        self.elementwise_op(rhs, |a, b| a + b)
    }

    fn elementwise_sub(&self, rhs: &Self) -> Result<Vec<T>, MathError> {
        self.elementwise_op(rhs, |a, b| a - b)
    }
}

impl<T> VecElementwiseArithmetic<T> for Vec<T> where T: Copy + Sub<Output = T> + Add<Output = T> {}
