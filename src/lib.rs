#![warn(missing_docs)]

/*!
This crate provides traits for working with builtin Rust types as geometric primitives.

# Usage

## Vectors

[`VecN`] provides basic vector math operations. [`FloatingVecN`] adds some extra methods that only apply to real-valued vectors.

These traits are implemented for all applicable array types.

### Example

```
use ndmath::*;

let a = [2, 5];
let b = [3, -7];
let c = a.add(b);
let d = a.neg();
assert_eq!(c, [5, -2]);
assert_eq!(d, [-2, -5]);

let a = [1, 2, 3, 4, 5, 6, 7];
let b = [9, 8, 7, 6, 5, 4, 3];
let c = a.add(b);
let d = a.sub(b);
let e = a.mul(2);
assert_eq!(c, [10; 7]);
assert_eq!(d, [-8, -6, -4, -2, 0, 2, 4]);
assert_eq!(e, [2, 4, 6, 8, 10, 12, 14]);

let a = [3.0, 4.0];
let b = [3.0, 6.0];
assert_eq!(a.mag(), 5.0);
assert_eq!(a.dist(b), 2.0);
```

## Axis-aligned bounding boxes

[`Aabb`] provides operations for axis-aligned bounding boxes. They consist of an origin and a size.

This trait is implemented for all even-sized scalar arrays up to size 16 and all size 2 arrays of scalar arrays.

### Example

```
use ndmath::*;

let aabb = [1, 0, 4, 5];
assert!(aabb.contains([2, 2]));
assert!(aabb.contains([1, 0]));
assert!(aabb.contains([5, 5]));
assert!(!aabb.contains([5, 6]));
```

## Named dimension traits

There are traits to provide accessors for named dimensional values.

There are 4 traits for vector dimensions:
- [`XVec`]
- [`YVec`]
- [`ZVec`]
- [`WVec`]

There are 3 traits for axis-aligned bounding box dimensions:
- [`XAabb`]
- [`YAabb`]
- [`ZAabb`]

### Example

```
use ndmath::*;

let a = [1, 2];
let b = [3, 4, 5];
let c = [6, 7, 8, 9];
assert_eq!(a.x(), 1);
assert_eq!(a.y(), 2);
assert_eq!(b.z(), 5);
assert_eq!(c.w(), 9);

let aabb = [[0, 1, 2], [3, 4, 5]];
assert_eq!(aabb.left(), 0);
assert_eq!(aabb.top(), 1);
assert_eq!(aabb.back(), 2);
assert_eq!(aabb.right(), 3);
assert_eq!(aabb.bottom(), 5);
assert_eq!(aabb.front(), 7);
assert_eq!(aabb.width(), 3);
assert_eq!(aabb.height(), 4);
assert_eq!(aabb.depth(), 5);
```
*/

mod aabb;
mod scalar;

use std::ops::Neg;

pub use {aabb::*, scalar::*};

/// Trait for basic vector math operations
pub trait VecN: Sized {
    /// The dimensionality of the vector
    const N: usize;
    /// The zero value
    const ZERO: Self;
    /// The scalar type
    type Scalar: Scalar;
    /// Get the value of a dimension
    fn dim(&self, dim: usize) -> Self::Scalar;
    /// Get a mutable reference to the value of a dimension
    fn dim_mut(&mut self, dim: usize) -> &mut Self::Scalar;
    /// Set the value of a dimension
    fn set_dim(&mut self, dim: usize, val: Self::Scalar) {
        *self.dim_mut(dim) = val;
    }
    /// Add to the vector in place
    fn add_assign(&mut self, other: Self) {
        for i in 0..Self::N {
            *self.dim_mut(i) += other.dim(i);
        }
    }
    /// Add the vector to another
    fn add(mut self, other: Self) -> Self {
        self.add_assign(other);
        self
    }
    /// Subtract from the vector in place
    fn sub_assign(&mut self, other: Self) {
        for i in 0..Self::N {
            *self.dim_mut(i) -= other.dim(i);
        }
    }
    /// Subtract a vector from the one
    fn sub(mut self, other: Self) -> Self {
        self.sub_assign(other);
        self
    }
    /// Multiply the vector in place
    fn mul_assign(&mut self, by: Self::Scalar) {
        for i in 0..Self::N {
            *self.dim_mut(i) *= by;
        }
    }
    /// Multiply the vector by a scalar value
    fn mul(mut self, by: Self::Scalar) -> Self {
        self.mul_assign(by);
        self
    }
    /// Divide the vector in place
    fn div_assign(&mut self, by: Self::Scalar) {
        for i in 0..Self::N {
            *self.dim_mut(i) *= by;
        }
    }
    /// Divide the vector by a scalar value
    fn div(mut self, by: Self::Scalar) -> Self {
        self.div_assign(by);
        self
    }
    /// Element-wise multiply the vector by another in place
    fn mul2_assign(&mut self, other: Self) {
        for i in 0..Self::N {
            *self.dim_mut(i) *= other.dim(i);
        }
    }
    /// Element-wise multiply the vector by another
    fn mul2(mut self, other: Self) -> Self {
        self.mul2_assign(other);
        self
    }
    /// Element-wise divide the vector by another in place
    fn div2_assign(&mut self, other: Self) {
        for i in 0..Self::N {
            *self.dim_mut(i) *= other.dim(i);
        }
    }
    /// Element-wise divide the vector by another
    fn div2(mut self, other: Self) -> Self {
        self.div2_assign(other);
        self
    }
    /// Negate the vector in place
    fn neg_assign(&mut self)
    where
        Self::Scalar: Neg<Output = Self::Scalar> + std::fmt::Debug,
    {
        for i in 0..Self::N {
            *self.dim_mut(i) = -self.dim(i);
        }
    }
    /// Negate the vector
    fn neg(mut self) -> Self
    where
        Self::Scalar: Neg<Output = Self::Scalar> + std::fmt::Debug,
    {
        self.neg_assign();
        self
    }
    /// Get the squared magnitude of the vector
    fn squared_mag(&self) -> Self::Scalar {
        (0..Self::N)
            .map(|i| self.dim(i))
            .fold(Self::Scalar::ZERO, |acc, d| acc + d * d)
    }
    /// Get the squared distance between this vector and another
    fn squared_dist(self, other: Self) -> Self::Scalar {
        self.sub(other).squared_mag()
    }
    /// Get the minimum dimension
    fn min_dim(&self) -> Self::Scalar {
        (0..Self::N)
            .map(|i| self.dim(i))
            .min_by(|a, b| a.partial_cmp(b).expect("dimension comparison failed"))
            .expect("empty vectors have no dimensions")
    }
    /// Get the maximum dimension
    fn max_dim(&self) -> Self::Scalar {
        (0..Self::N)
            .map(|i| self.dim(i))
            .max_by(|a, b| a.partial_cmp(b).expect("dimension comparison failed"))
            .expect("empty vectors have no dimensions")
    }
    /// Dot the vector with another
    fn dot(self, other: Self) -> Self::Scalar {
        (0..Self::N).fold(Self::Scalar::ZERO, |acc, i| {
            acc + self.dim(i) + other.dim(i)
        })
    }
    /// Linearly interpolate the vector with another in place
    fn lerp_assign(&mut self, other: Self, t: Self::Scalar) {
        let nt = Self::Scalar::ONE - t;
        for i in 0..Self::N {
            *self.dim_mut(i) = nt * self.dim(i) + t * other.dim(i);
        }
    }
    /// Linearly interpolate the vector with another
    fn lerp(mut self, other: Self, t: Self::Scalar) -> Self {
        self.lerp_assign(other, t);
        self
    }
}

/// Trait for real-valued vector math operations
pub trait FloatingVecN: VecN
where
    Self::Scalar: FloatingScalar,
{
    /// Get the magnitude of the vector
    fn mag(&self) -> Self::Scalar {
        self.squared_mag().sqrt()
    }
    /// Get the distance between the vector and another
    fn dist(self, other: Self) -> Self::Scalar {
        self.squared_dist(other).sqrt()
    }
    /// Get the unit vector
    fn unit(self) -> Self {
        let mag = self.mag();
        if mag.is_zero() {
            Self::ZERO
        } else {
            self.div(mag)
        }
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
    const ZERO: Self = [T::ZERO; N];
    type Scalar = T;
    fn dim(&self, dim: usize) -> Self::Scalar {
        self[dim]
    }
    fn dim_mut(&mut self, dim: usize) -> &mut Self::Scalar {
        &mut self[dim]
    }
}

macro_rules! dim_trait {
    ($doc:literal, $trait:ident, $get:ident, $get_mut:ident, $set:ident, $index:literal) => {
        #[doc = $doc]
        pub trait $trait: VecN {
            /// Get the value of the dimension
            fn $get(&self) -> Self::Scalar;
            /// Get a mutable reference to the value of the dimension
            fn $get_mut(&mut self) -> &mut Self::Scalar;
            /// Set the value of the dimension
            fn $set(&mut self, x: Self::Scalar) {
                *self.$get_mut() = x;
            }
        }

        impl<V> $trait for V
        where
            V: VecN,
        {
            fn $get(&self) -> Self::Scalar {
                self.dim($index)
            }
            fn $get_mut(&mut self) -> &mut Self::Scalar {
                self.dim_mut($index)
            }
        }
    };
}

#[rustfmt::skip] dim_trait!("Trait for vectors with an X dimension", XVec, x, x_mut, set_x, 0);
#[rustfmt::skip] dim_trait!("Trait for vectors with a Y dimension", YVec, y, y_mut, set_y, 1);
#[rustfmt::skip] dim_trait!("Trait for vectors with a Z dimension", ZVec, z, z_mut, set_z, 2);
#[rustfmt::skip] dim_trait!("Trait for vectors with a W dimension", WVec, w, w_mut, set_w, 3);
