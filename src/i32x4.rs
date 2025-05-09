use crate::bx4::Bx4;
use crate::{impl_vec_cmp, impl_vec_op, impl_vec_unary_op};
use core::arch::wasm32::{
    i32x4, i32x4_add, i32x4_all_true, i32x4_extract_lane, i32x4_mul, i32x4_neg, i32x4_replace_lane,
    i32x4_shl, i32x4_shr, i32x4_shuffle, i32x4_splat, i32x4_sub, v128, v128_bitselect,
};
use std::fmt::Debug;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Shl, ShlAssign, Shr, ShrAssign, Sub,
    SubAssign,
};

macro_rules! impl_vec_overload_op {
    ($trait:ident, $fn:ident, $op_fn:ident) => {
        impl $trait for I32x4 {
            type Output = Self;
            #[inline]
            fn $fn(self, other: Self) -> Self::Output {
                Self {
                    data: $op_fn(self.data, other.data),
                }
            }
        }

        impl $trait<i32> for I32x4 {
            type Output = Self;
            #[inline]
            fn $fn(self, other: i32) -> Self::Output {
                Self {
                    data: $op_fn(self.data, i32x4_splat(other)),
                }
            }
        }

        impl $trait<I32x4> for i32 {
            type Output = I32x4;
            #[inline]
            fn $fn(self, other: I32x4) -> Self::Output {
                I32x4 {
                    data: $op_fn(i32x4_splat(self), other.data),
                }
            }
        }
    };
}

macro_rules! impl_vec_assign_op {
    ($trait:ident, $fn:ident, $op:tt) => {
        impl $trait for I32x4 {
            #[inline]
            fn $fn(&mut self, other: Self) {
                (*self) = self.clone() $op other;
            }
        }

        impl $trait<i32> for I32x4 {
            #[inline]
            fn $fn(&mut self, other: i32) {
                (*self) = self.clone() $op Self::splat(other);
            }
        }
    };
}

pub struct I32x4 {
    data: v128,
}

impl I32x4 {
    pub fn default() -> Self {
        Self {
            data: i32x4(0, 0, 0, 0),
        }
    }

    pub fn new(v1: i32, v2: i32, v3: i32, v4: i32) -> Self {
        Self {
            data: i32x4(v1, v2, v3, v4),
        }
    }

    pub fn splat(value: i32) -> Self {
        Self {
            data: i32x4_splat(value),
        }
    }

    pub fn new_from_fn<F>(f: F) -> Self
    where
        F: Fn(usize) -> i32,
    {
        Self {
            data: i32x4(f(0), f(1), f(2), f(3)),
        }
    }

    impl_vec_cmp!(eq, s_eq, i32x4_eq, Bx4);
    impl_vec_cmp!(ne, s_ne, i32x4_ne, Bx4);
    impl_vec_cmp!(lt, s_lt, i32x4_lt, Bx4);
    impl_vec_cmp!(le, s_le, i32x4_le, Bx4);
    impl_vec_cmp!(gt, s_gt, i32x4_gt, Bx4);
    impl_vec_cmp!(ge, s_ge, i32x4_ge, Bx4);

    impl_vec_op!(min, s_min, i32x4_min, i32);
    impl_vec_op!(max, s_max, i32x4_max, i32);

    impl_vec_unary_op!(abs, i32x4_abs);

    pub fn apply<F>(&self, f: F) -> Self
    where
        F: Fn(i32) -> i32,
    {
        let (v1, v2, v3, v4) = self.extract_lanes();
        Self::new(f(v1), f(v2), f(v3), f(v4))
    }

    pub fn reduce<F>(&self, f: F, init_val: i32) -> i32
    where
        F: Fn(i32, i32) -> i32,
    {
        let (v1, v2, v3, v4) = self.extract_lanes();
        f(f(f(f(init_val, v1), v2), v3), v4)
    }

    pub fn reduce_add(&self) -> i32 {
        self.reduce(|a, b| a + b, 0)
    }

    pub fn reduce_mul(&self) -> i32 {
        self.reduce(|a, b| a * b, 1)
    }

    pub fn reduce_min(&self) -> i32 {
        self.reduce(|a, b| if a < b { a } else { b }, i32::MAX)
    }

    pub fn reduce_max(&self) -> i32 {
        self.reduce(|a, b| if a > b { a } else { b }, i32::MIN)
    }

    pub fn extract_lanes(&self) -> (i32, i32, i32, i32) {
        (
            i32x4_extract_lane::<0>(self.data),
            i32x4_extract_lane::<1>(self.data),
            i32x4_extract_lane::<2>(self.data),
            i32x4_extract_lane::<3>(self.data),
        )
    }

    pub fn extract_lane(&self, index: usize) -> i32 {
        match index {
            0 => i32x4_extract_lane::<0>(self.data),
            1 => i32x4_extract_lane::<1>(self.data),
            2 => i32x4_extract_lane::<2>(self.data),
            3 => i32x4_extract_lane::<3>(self.data),
            _ => panic!("Index out of bounds"),
        }
    }

    pub fn set_lane(&mut self, index: usize, value: i32) {
        let new_vec = match index {
            0 => i32x4_replace_lane::<0>(self.data, value),
            1 => i32x4_replace_lane::<1>(self.data, value),
            2 => i32x4_replace_lane::<2>(self.data, value),
            3 => i32x4_replace_lane::<3>(self.data, value),
            _ => panic!("Index out of bounds"),
        };

        self.data = new_vec;
    }

    pub fn if_else(self, other: Self, mask: Bx4) -> Self {
        let data = v128_bitselect(self.data, other.data, mask.data);
        Self { data }
    }

    pub fn all_nonzero(self) -> bool {
        i32x4_all_true(self.data)
    }

    pub fn shuffle<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
        self,
        other: Self,
    ) -> Self {
        let data = i32x4_shuffle::<I0, I1, I2, I3>(self.data, other.data);
        Self { data }
    }
}

impl Clone for I32x4 {
    fn clone(&self) -> Self {
        Self { data: self.data }
    }
}

impl Into<[i32; 4]> for I32x4 {
    fn into(self) -> [i32; 4] {
        let (v1, v2, v3, v4) = self.extract_lanes();
        [v1, v2, v3, v4]
    }
}

impl Into<Vec<i32>> for I32x4 {
    fn into(self) -> Vec<i32> {
        let (v1, v2, v3, v4) = self.extract_lanes();
        vec![v1, v2, v3, v4]
    }
}

impl From<F32x4> for I32x4 {
    fn from(value: F32x4) -> Self {
        Self {
            data: i32x4_trunc_sat_f32x4(value.data),
        }
    }
}

impl From<[i32; 4]> for I32x4 {
    fn from(arr: [i32; 4]) -> Self {
        let [v1, v2, v3, v4] = arr;
        Self::new(v1, v2, v3, v4)
    }
}

impl From<&[i32; 4]> for I32x4 {
    fn from(arr: &[i32; 4]) -> Self {
        let [v1, v2, v3, v4] = *arr;
        Self::new(v1, v2, v3, v4)
    }
}

impl Neg for I32x4 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            data: i32x4_neg(self.data),
        }
    }
}

impl Debug for I32x4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (v1, v2, v3, v4) = self.extract_lanes();
        f.debug_tuple("I32x4")
            .field(&v1)
            .field(&v2)
            .field(&v3)
            .field(&v4)
            .finish()
    }
}

impl_vec_overload_op!(Add, add, i32x4_add);
impl_vec_overload_op!(Sub, sub, i32x4_sub);
impl_vec_overload_op!(Mul, mul, i32x4_mul);

// There is no native division operation in WASM SIMD, so we implement it using by diving
// each value lane-wise
impl Div for I32x4 {
    type Output = Self;
    #[inline]
    fn div(self, other: Self) -> Self::Output {
        let (n1, n2, n3, n4) = self.extract_lanes();
        let (d1, d2, d3, d4) = other.extract_lanes();
        Self::new(n1 / d1, n2 / d2, n3 / d3, n4 / d4)
    }
}

impl Div<i32> for I32x4 {
    type Output = Self;
    #[inline]
    fn div(self, other: i32) -> Self::Output {
        let (n1, n2, n3, n4) = self.extract_lanes();
        Self::new(n1 / other, n2 / other, n3 / other, n4 / other)
    }
}

impl Div<I32x4> for i32 {
    type Output = I32x4;
    #[inline]
    fn div(self, other: I32x4) -> Self::Output {
        let (n1, n2, n3, n4) = other.extract_lanes();
        I32x4::new(self / n1, self / n2, self / n3, self / n4)
    }
}

impl Shl<u32> for I32x4 {
    type Output = Self;
    #[inline]
    fn shl(self, amt: u32) -> Self::Output {
        Self {
            data: i32x4_shl(self.data, amt),
        }
    }
}

impl Shr<u32> for I32x4 {
    type Output = Self;
    #[inline]
    fn shr(self, amt: u32) -> Self::Output {
        Self {
            data: i32x4_shr(self.data, amt),
        }
    }
}

impl_vec_assign_op!(AddAssign, add_assign, +);
impl_vec_assign_op!(SubAssign, sub_assign, -);
impl_vec_assign_op!(MulAssign, mul_assign, *);
impl_vec_assign_op!(DivAssign, div_assign, /);

impl ShlAssign<u32> for I32x4 {
    #[inline]
    fn shl_assign(&mut self, amt: u32) {
        self.data = i32x4_shl(self.data, amt);
    }
}

impl ShrAssign<u32> for I32x4 {
    #[inline]
    fn shr_assign(&mut self, amt: u32) {
        self.data = i32x4_shr(self.data, amt);
    }
}
