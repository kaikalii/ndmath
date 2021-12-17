use crate::{Scalar, VecN};

/// Trait for axis-aligned bounding boxes
pub trait Aabb: Sized {
    /// The vector type
    type Vector: VecN;
    /// The origin-aligned aabb with 0 volume
    const ORIGIN_ZERO_SIZE: Self;
    /// Get the origin value of a dimension
    fn origin_dim(&self, dim: usize) -> <Self::Vector as VecN>::Scalar;
    /// Get a mutable reference to the origin value of a dimension
    fn origin_dim_mut(&mut self, dim: usize) -> &mut <Self::Vector as VecN>::Scalar;
    /// Get the size value of a dimension
    fn size_dim(&self, dim: usize) -> <Self::Vector as VecN>::Scalar;
    /// Get a mutable reference to the size value of a dimension
    fn size_dim_mut(&mut self, dim: usize) -> &mut <Self::Vector as VecN>::Scalar;
    /// Set the origin value of a dimension
    fn set_origin_dim(&mut self, dim: usize, val: <Self::Vector as VecN>::Scalar) {
        *self.origin_dim_mut(dim) = val;
    }
    /// Set the size value of a dimension
    fn set_size_dim(&mut self, dim: usize, val: <Self::Vector as VecN>::Scalar) {
        *self.size_dim_mut(dim) = val;
    }
    /// Get the end value of a dimension
    fn end_dim(&self, dim: usize) -> <Self::Vector as VecN>::Scalar {
        self.origin_dim(dim) + self.size_dim(dim)
    }
    /// The the center value of a dimension
    fn center_dim(&self, dim: usize) -> <Self::Vector as VecN>::Scalar {
        self.origin_dim(dim) + self.size_dim(dim) / <Self::Vector as VecN>::Scalar::TWO
    }
    /// The center of the aabb
    fn center(&self) -> Self::Vector {
        let mut v = Self::Vector::ZERO;
        for i in 0..Self::Vector::N {
            v.set_dim(i, self.center_dim(i));
        }
        v
    }
    /// Check if the aabb contains a vector
    fn contains(&self, v: Self::Vector) -> bool {
        for i in 0..Self::Vector::N {
            let d = v.dim(i);
            let origin = self.origin_dim(i);
            if d < origin || d > origin + self.size_dim(i) {
                return false;
            }
        }
        true
    }
    /// Get the aabb the bounds a list of vectors
    fn bounding<I>(iter: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Vector>,
        Self::Vector: Clone,
    {
        let mut iter = iter.into_iter();
        let mut min = iter.next()?;
        let mut max = min.clone();
        for v in iter {
            for i in 0..Self::Vector::N {
                let d = v.dim(i);
                if d < min.dim(i) {
                    min.set_dim(i, d);
                } else if d > max.dim(i) {
                    max.set_dim(i, d);
                }
            }
        }
        let mut res = Self::ORIGIN_ZERO_SIZE;
        for i in 0..Self::Vector::N {
            let min = min.dim(i);
            res.set_origin_dim(i, min);
            res.set_size_dim(i, max.dim(i) - min);
        }
        Some(res)
    }
}

impl<T, const N: usize> Aabb for [[T; N]; 2]
where
    T: Scalar,
{
    type Vector = [T; N];
    const ORIGIN_ZERO_SIZE: Self = [[<Self::Vector as VecN>::Scalar::ZERO; N]; 2];
    fn origin_dim(&self, dim: usize) -> <Self::Vector as VecN>::Scalar {
        self[0][dim]
    }
    fn origin_dim_mut(&mut self, dim: usize) -> &mut <Self::Vector as VecN>::Scalar {
        &mut self[0][dim]
    }
    fn size_dim(&self, dim: usize) -> <Self::Vector as VecN>::Scalar {
        self[1][dim]
    }
    fn size_dim_mut(&mut self, dim: usize) -> &mut <Self::Vector as VecN>::Scalar {
        &mut self[1][dim]
    }
}

macro_rules! aabb_impl {
    ($($size:literal),* $(,)?) => {
        $(
            impl<T> Aabb for [T; $size * 2]
            where
                T: Scalar,
            {
                type Vector = [T; $size];
                const ORIGIN_ZERO_SIZE: Self = [<Self::Vector as VecN>::Scalar::ZERO; $size * 2];
                fn origin_dim(&self, dim: usize) -> <Self::Vector as VecN>::Scalar {
                    self[dim]
                }
                fn origin_dim_mut(&mut self, dim: usize) -> &mut <Self::Vector as VecN>::Scalar {
                    &mut self[dim]
                }
                fn size_dim(&self, dim: usize) -> <Self::Vector as VecN>::Scalar {
                    self[Self::Vector::N + dim]
                }
                fn size_dim_mut(&mut self, dim: usize) -> &mut <Self::Vector as VecN>::Scalar {
                    &mut self[Self::Vector::N + dim]
                }
            }
        )*
    };
}

aabb_impl!(1, 2, 3, 4, 5, 6, 7, 8);

macro_rules! dim_trait {
    (
        $doc:literal,
        $trait:ident,
        $get_origin:ident,
        $get_origin_mut:ident,
        $set_origin:ident,
        $get_size:ident,
        $get_size_mut:ident,
        $set_size:ident,
        $get_end:ident,
        $index:literal
    ) => {
        #[doc = $doc]
        pub trait $trait: Aabb {
            /// Get the origin value of the dimension
            fn $get_origin(&self) -> <Self::Vector as VecN>::Scalar;
            /// Get a mutable reference to the origin value of the dimension
            fn $get_origin_mut(&mut self) -> &mut <Self::Vector as VecN>::Scalar;
            /// Get the size value of the dimension
            fn $get_size(&self) -> <Self::Vector as VecN>::Scalar;
            /// Get a mutable reference to the size value of the dimension
            fn $get_size_mut(&mut self) -> &mut <Self::Vector as VecN>::Scalar;
            /// Set the origin value of the dimension
            fn $set_origin(&mut self, x: <Self::Vector as VecN>::Scalar) {
                *self.$get_origin_mut() = x;
            }
            /// Set the size value of the dimension
            fn $set_size(&mut self, x: <Self::Vector as VecN>::Scalar) {
                *self.$get_size_mut() = x;
            }
            /// Get the end value of the dimension
            fn $get_end(&self) -> <Self::Vector as VecN>::Scalar {
                self.end_dim($index)
            }
        }

        impl<A> $trait for A
        where
            A: Aabb,
        {
            fn $get_origin(&self) -> <Self::Vector as VecN>::Scalar {
                self.origin_dim($index)
            }
            fn $get_origin_mut(&mut self) -> &mut <Self::Vector as VecN>::Scalar {
                self.origin_dim_mut($index)
            }
            fn $get_size(&self) -> <Self::Vector as VecN>::Scalar {
                self.size_dim($index)
            }
            fn $get_size_mut(&mut self) -> &mut <Self::Vector as VecN>::Scalar {
                self.size_dim_mut($index)
            }
        }
    };
}

dim_trait!(
    "Trait for axis-aligned bounding boxes with a width",
    XAabb,
    left,
    left_mut,
    set_left,
    width,
    width_mut,
    set_width,
    right,
    0
);
dim_trait!(
    "Trait for axis-aligned bounding boxes with a height",
    YAabb,
    top,
    top_mut,
    set_top,
    height,
    height_mut,
    set_height,
    bottom,
    1
);
dim_trait!(
    "Trait for axis-aligned bounding boxes with a depth",
    ZAabb,
    back,
    back_mut,
    set_back,
    depth,
    depth_mut,
    set_depth,
    front,
    2
);
