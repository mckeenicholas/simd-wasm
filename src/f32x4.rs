use crate::bx4::Bx4;
use crate::{impl_vec_cmp, impl_vec_op, impl_vec_unary_op};
use core::arch::wasm32::*;

pub struct F32x4 {
    data: v128,
}

impl F32x4 {
    pub fn default() -> Self {
        Self {
            data: f32x4(0.0, 0.0, 0.0, 0.0),
        }
    }

    pub fn new(v1: f32, v2: f32, v3: f32, v4: f32) -> Self {
        Self {
            data: f32x4(v1, v2, v3, v4),
        }
    }

    #[inline]
    pub fn splat(value: f32) -> Self {
        Self {
            data: f32x4_splat(value),
        }
    }

    pub fn new_from_fn<F>(f: F) -> Self
    where
        F: Fn(usize) -> f32,
    {
        Self {
            data: f32x4(f(0), f(1), f(2), f(3)),
        }
    }

    impl_vec_cmp!(eq, s_eq, f32x4_eq, Bx4);
    impl_vec_cmp!(ne, s_ne, f32x4_ne, Bx4);

    impl_vec_cmp!(lt, s_lt, f32x4_lt, Bx4);
    impl_vec_cmp!(le, s_le, f32x4_le, Bx4);
    impl_vec_cmp!(gt, s_gt, f32x4_gt, Bx4);
    impl_vec_cmp!(ge, s_ge, f32x4_ge, Bx4);

    impl_vec_op!(min, s_min, f32x4_min, f32);
    impl_vec_op!(max, s_max, f32x4_max, f32);

    impl_vec_unary_op!(abs, f32x4_abs);
}

impl From<I32x4> for F32x4 {
    
}