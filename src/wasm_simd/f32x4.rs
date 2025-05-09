use crate::wasm_simd::bx4::Bx4;
use crate::wasm_simd::i32x4::I32x4;
use crate::{
    impl_debug, impl_default, impl_vec_assign_op, impl_vec_binary_op, impl_vec_cmp,
    impl_vec_overload_op, impl_vec_unary_op,
};
use core::arch::wasm32::*;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub struct F32x4(v128);

impl F32x4 {
    pub fn new(v1: f32, v2: f32, v3: f32, v4: f32) -> Self {
        Self(f32x4(v1, v2, v3, v4))
    }

    pub fn splat(value: f32) -> Self {
        Self(f32x4_splat(value))
    }

    pub fn new_from_fn<F>(f: F) -> Self
    where
        F: Fn(usize) -> f32,
    {
        Self(f32x4(f(0), f(1), f(2), f(3)))
    }

    pub(crate) fn from_v128(data: v128) -> Self {
        Self(data)
    }

    pub(crate) fn to_v128(self) -> v128 {
        self.0
    }

    pub fn apply<F>(&self, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        let (v1, v2, v3, v4) = self.extract_lanes();
        Self::new(f(v1), f(v2), f(v3), f(v4))
    }

    pub fn extract_lanes(&self) -> (f32, f32, f32, f32) {
        (
            f32x4_extract_lane::<0>(self.0),
            f32x4_extract_lane::<1>(self.0),
            f32x4_extract_lane::<2>(self.0),
            f32x4_extract_lane::<3>(self.0),
        )
    }

    pub fn extract_lane(&self, index: usize) -> f32 {
        match index {
            0 => f32x4_extract_lane::<0>(self.0),
            1 => f32x4_extract_lane::<1>(self.0),
            2 => f32x4_extract_lane::<2>(self.0),
            3 => f32x4_extract_lane::<3>(self.0),
            _ => panic!("Index out of bounds"),
        }
    }

    pub fn set_lane(&mut self, index: usize, value: f32) {
        let new_vec = match index {
            0 => f32x4_replace_lane::<0>(self.0, value),
            1 => f32x4_replace_lane::<1>(self.0, value),
            2 => f32x4_replace_lane::<2>(self.0, value),
            3 => f32x4_replace_lane::<3>(self.0, value),
            _ => panic!("Index out of bounds"),
        };

        self.0 = new_vec;
    }

    pub fn if_else(self, other: &Self, mask: Bx4) -> Self {
        let data = v128_bitselect(self.0, other.0, mask.to_v128());
        Self(data)
    }

    impl_vec_cmp!(eq, s_eq, f32x4_eq, Bx4);
    impl_vec_cmp!(ne, s_ne, f32x4_ne, Bx4);

    impl_vec_cmp!(lt, s_lt, f32x4_lt, Bx4);
    impl_vec_cmp!(le, s_le, f32x4_le, Bx4);
    impl_vec_cmp!(gt, s_gt, f32x4_gt, Bx4);
    impl_vec_cmp!(ge, s_ge, f32x4_ge, Bx4);

    impl_vec_binary_op!(min, s_min, f32x4_min, f32);
    impl_vec_binary_op!(max, s_max, f32x4_max, f32);

    impl_vec_unary_op!(abs, f32x4_abs);
    impl_vec_unary_op!(ceil, f32x4_ceil);
    impl_vec_unary_op!(floor, f32x4_floor);
}

impl_default!(F32x4, f32);

impl Clone for F32x4 {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for F32x4 {}

impl From<I32x4> for F32x4 {
    fn from(value: I32x4) -> Self {
        Self(f32x4_convert_i32x4(value.to_v128()))
    }
}

impl From<[f32; 4]> for F32x4 {
    fn from(arr: [f32; 4]) -> Self {
        let [v1, v2, v3, v4] = arr;
        Self::new(v1, v2, v3, v4)
    }
}
impl From<F32x4> for [f32; 4] {
    fn from(val: F32x4) -> Self {
        let (v1, v2, v3, v4) = val.extract_lanes();
        [v1, v2, v3, v4]
    }
}

impl From<F32x4> for Vec<f32> {
    fn from(val: F32x4) -> Self {
        let (v1, v2, v3, v4) = val.extract_lanes();
        vec![v1, v2, v3, v4]
    }
}

impl_debug!(F32x4, (v1, v2, v3, v4));

impl Neg for F32x4 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(i32x4_neg(self.0))
    }
}

impl_vec_overload_op!(F32x4, f32, Add, add, f32x4_add);
impl_vec_overload_op!(F32x4, f32, Sub, sub, f32x4_sub);
impl_vec_overload_op!(F32x4, f32, Mul, mul, f32x4_mul);
impl_vec_overload_op!(F32x4, f32, Div, div, f32x4_div);

impl_vec_assign_op!(F32x4, f32, AddAssign, add_assign, +);
impl_vec_assign_op!(F32x4, f32, SubAssign, sub_assign, -);
impl_vec_assign_op!(F32x4, f32, MulAssign, mul_assign, *);
impl_vec_assign_op!(F32x4, f32, DivAssign, div_assign, /);
