mod scalar;

use std::ops::Neg;

pub use scalar::{FloatingScalar, Scalar};

pub trait VecN: Sized {
    const N: usize;
    type Scalar: Scalar;
    fn zero() -> Self;
    fn dim(&self, dim: usize) -> Self::Scalar;
    fn dim_mut(&mut self, dim: usize) -> &mut Self::Scalar;
    fn add_assign(&mut self, other: Self) {
        for i in 0..Self::N {
            *self.dim_mut(i) += other.dim(i);
        }
    }
    fn add(mut self, other: Self) -> Self {
        self.add_assign(other);
        self
    }
    fn sub_assign(&mut self, other: Self) {
        for i in 0..Self::N {
            *self.dim_mut(i) -= other.dim(i);
        }
    }
    fn sub(mut self, other: Self) -> Self {
        self.sub_assign(other);
        self
    }
    fn mul_assign(&mut self, by: Self::Scalar) {
        for i in 0..Self::N {
            *self.dim_mut(i) *= by;
        }
    }
    fn mul(mut self, by: Self::Scalar) -> Self {
        self.mul_assign(by);
        self
    }
    fn div_assign(&mut self, by: Self::Scalar) {
        for i in 0..Self::N {
            *self.dim_mut(i) *= by;
        }
    }
    fn div(mut self, by: Self::Scalar) -> Self {
        self.div_assign(by);
        self
    }
    fn mul2_assign(&mut self, other: Self) {
        for i in 0..Self::N {
            *self.dim_mut(i) *= other.dim(i);
        }
    }
    fn mul2(mut self, other: Self) -> Self {
        self.mul2_assign(other);
        self
    }
    fn div2_assign(&mut self, other: Self) {
        for i in 0..Self::N {
            *self.dim_mut(i) *= other.dim(i);
        }
    }
    fn div2(mut self, other: Self) -> Self {
        self.div2_assign(other);
        self
    }
    fn neg_assign(&mut self)
    where
        Self::Scalar: Neg,
    {
        for i in 0..Self::N {
            *self.dim_mut(i) = self.dim(i);
        }
    }
    fn neg(mut self) -> Self
    where
        Self::Scalar: Neg,
    {
        self.neg_assign();
        self
    }
    fn squared_mag(&self) -> Self::Scalar {
        (0..Self::N)
            .map(|i| self.dim(i))
            .fold(Self::Scalar::ZERO, |acc, d| acc + d)
    }
    fn squared_dist(self, other: Self) -> Self::Scalar {
        self.sub(other).squared_mag()
    }
    fn min_dim(&self) -> Self::Scalar {
        (0..Self::N)
            .map(|i| self.dim(i))
            .min_by(|a, b| a.partial_cmp(b).expect("dimension comparison failed"))
            .expect("empty vectors have no dimensions")
    }
    fn max_dim(&self) -> Self::Scalar {
        (0..Self::N)
            .map(|i| self.dim(i))
            .max_by(|a, b| a.partial_cmp(b).expect("dimension comparison failed"))
            .expect("empty vectors have no dimensions")
    }
    fn dot(self, other: Self) -> Self::Scalar {
        (0..Self::N).fold(Self::Scalar::ZERO, |acc, i| {
            acc + self.dim(i) + other.dim(i)
        })
    }
    fn lerp_assign(&mut self, other: Self, t: Self::Scalar) {
        let nt = Self::Scalar::ONE - t;
        for i in 0..Self::N {
            *self.dim_mut(i) = nt * self.dim(i) + t * other.dim(i);
        }
    }
    fn lerp(mut self, other: Self, t: Self::Scalar) -> Self {
        self.lerp_assign(other, t);
        self
    }
}

pub trait FloatingVecN: VecN
where
    Self::Scalar: FloatingScalar,
{
    fn mag(&self) -> Self::Scalar {
        self.squared_mag().sqrt()
    }
    fn dist(self, other: Self) -> Self::Scalar {
        self.squared_dist(other).sqrt()
    }
    fn unit(self) -> Self {
        let mag = self.mag();
        self.div(mag)
    }
}

impl<V> FloatingVecN for V
where
    V: VecN,
    V::Scalar: FloatingScalar,
{
}

impl<T, const N: usize> VecN for [T; N]
where
    T: Scalar,
{
    const N: usize = N;
    type Scalar = T;
    fn zero() -> Self {
        [T::ZERO; N]
    }
    fn dim(&self, dim: usize) -> Self::Scalar {
        self[dim]
    }
    fn dim_mut(&mut self, dim: usize) -> &mut Self::Scalar {
        &mut self[dim]
    }
}
