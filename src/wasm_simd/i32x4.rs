use crate::wasm_simd::bx4::Bx4;
use crate::wasm_simd::f32x4::F32x4;
use crate::{
    impl_debug, impl_default, impl_vec_assign_op, impl_vec_binary_op, impl_vec_cmp,
    impl_vec_overload_op, impl_vec_unary_op,
};
use core::arch::wasm32::*;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Shl, ShlAssign, Shr, ShrAssign, Sub,
    SubAssign,
};

pub struct I32x4(pub(crate) v128);

impl I32x4 {
    pub fn new(v1: i32, v2: i32, v3: i32, v4: i32) -> Self {
        Self(i32x4(v1, v2, v3, v4))
    }

    pub fn splat(value: i32) -> Self {
        Self(i32x4_splat(value))
    }

    pub fn new_from_fn<F>(f: F) -> Self
    where
        F: Fn(usize) -> i32,
    {
        Self(i32x4(f(0), f(1), f(2), f(3)))
    }

    pub(crate) fn from_v128(data: v128) -> Self {
        Self(data)
    }

    pub(crate) fn to_v128(self) -> v128 {
        self.0
    }

    pub fn apply<F>(&self, f: F) -> Self
    where
        F: Fn(i32) -> i32,
    {
        let (v1, v2, v3, v4) = self.extract_lanes();
        Self::new(f(v1), f(v2), f(v3), f(v4))
    }

    pub fn fold<F>(&self, f: F, init_val: i32) -> i32
    where
        F: Fn(i32, i32) -> i32,
    {
        let (v1, v2, v3, v4) = self.extract_lanes();
        f(f(f(f(init_val, v1), v2), v3), v4)
    }

    pub fn reduce_add(&self) -> i32 {
        self.fold(|a, b| a + b, 0)
    }

    pub fn reduce_mul(&self) -> i32 {
        self.fold(|a, b| a * b, 1)
    }

    pub fn reduce_min(&self) -> i32 {
        self.fold(|a, b| if a < b { a } else { b }, i32::MAX)
    }

    pub fn reduce_max(&self) -> i32 {
        self.fold(|a, b| if a > b { a } else { b }, i32::MIN)
    }

    pub fn extract_lanes(&self) -> (i32, i32, i32, i32) {
        (
            i32x4_extract_lane::<0>(self.0),
            i32x4_extract_lane::<1>(self.0),
            i32x4_extract_lane::<2>(self.0),
            i32x4_extract_lane::<3>(self.0),
        )
    }

    pub fn extract_lane(&self, index: usize) -> i32 {
        match index {
            0 => i32x4_extract_lane::<0>(self.0),
            1 => i32x4_extract_lane::<1>(self.0),
            2 => i32x4_extract_lane::<2>(self.0),
            3 => i32x4_extract_lane::<3>(self.0),
            _ => panic!("Index out of bounds"),
        }
    }

    pub fn set_lane(&mut self, index: usize, value: i32) {
        let new_vec = match index {
            0 => i32x4_replace_lane::<0>(self.0, value),
            1 => i32x4_replace_lane::<1>(self.0, value),
            2 => i32x4_replace_lane::<2>(self.0, value),
            3 => i32x4_replace_lane::<3>(self.0, value),
            _ => panic!("Index out of bounds"),
        };

        self.0 = new_vec;
    }

    pub fn if_else(&self, other: &Self, mask: &Bx4) -> Self {
        let data = v128_bitselect(self.0, other.0, mask.to_v128());
        Self(data)
    }

    pub fn all_nonzero(self) -> bool {
        i32x4_all_true(self.0)
    }

    pub fn shuffle<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
        &self,
        other: &Self,
    ) -> Self {
        let data = i32x4_shuffle::<I0, I1, I2, I3>(self.0, other.0);
        Self(data)
    }

    impl_vec_cmp!(eq, s_eq, i32x4_eq, Bx4);
    impl_vec_cmp!(ne, s_ne, i32x4_ne, Bx4);
    impl_vec_cmp!(lt, s_lt, i32x4_lt, Bx4);
    impl_vec_cmp!(le, s_le, i32x4_le, Bx4);
    impl_vec_cmp!(gt, s_gt, i32x4_gt, Bx4);
    impl_vec_cmp!(ge, s_ge, i32x4_ge, Bx4);

    impl_vec_binary_op!(min, s_min, i32x4_min, i32);
    impl_vec_binary_op!(max, s_max, i32x4_max, i32);

    impl_vec_unary_op!(abs, i32x4_abs);
}

impl_default!(I32x4, i32);

impl Clone for I32x4 {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for I32x4 {}

impl From<I32x4> for [i32; 4] {
    fn from(val: I32x4) -> Self {
        let (v1, v2, v3, v4) = val.extract_lanes();
        [v1, v2, v3, v4]
    }
}

impl From<I32x4> for Vec<i32> {
    fn from(val: I32x4) -> Self {
        let (v1, v2, v3, v4) = val.extract_lanes();
        vec![v1, v2, v3, v4]
    }
}

impl From<F32x4> for I32x4 {
    fn from(value: F32x4) -> Self {
        Self(i32x4_trunc_sat_f32x4(value.to_v128()))
    }
}

impl From<[i32; 4]> for I32x4 {
    fn from(arr: [i32; 4]) -> Self {
        let [v1, v2, v3, v4] = arr;
        Self::new(v1, v2, v3, v4)
    }
}

impl_debug!(I32x4, (v1, v2, v3, v4));

impl Neg for I32x4 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(i32x4_neg(self.0))
    }
}

impl_vec_overload_op!(I32x4, i32, Add, add, i32x4_add);
impl_vec_overload_op!(I32x4, i32, Sub, sub, i32x4_sub);
impl_vec_overload_op!(I32x4, i32, Mul, mul, i32x4_mul);

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
        Self(i32x4_shl(self.0, amt))
    }
}

impl Shr<u32> for I32x4 {
    type Output = Self;
    #[inline]
    fn shr(self, amt: u32) -> Self::Output {
        Self(i32x4_shr(self.0, amt))
    }
}

impl_vec_assign_op!(I32x4, i32, AddAssign, add_assign, +);
impl_vec_assign_op!(I32x4, i32, SubAssign, sub_assign, -);
impl_vec_assign_op!(I32x4, i32, MulAssign, mul_assign, *);
impl_vec_assign_op!(I32x4, i32, DivAssign, div_assign, /);

impl ShlAssign<u32> for I32x4 {
    #[inline]
    fn shl_assign(&mut self, amt: u32) {
        self.0 = i32x4_shl(self.0, amt);
    }
}

impl ShrAssign<u32> for I32x4 {
    #[inline]
    fn shr_assign(&mut self, amt: u32) {
        self.0 = i32x4_shr(self.0, amt);
    }
}
