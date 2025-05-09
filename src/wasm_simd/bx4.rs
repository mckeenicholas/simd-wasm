use core::arch::wasm32::*;
use std::fmt::Debug;
use std::ops::Not;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign};

use crate::impl_default;

const BIT_MASK_32: i32 = -1i32;

pub struct Bx4(v128);

impl Bx4 {
    pub fn new(v1: bool, v2: bool, v3: bool, v4: bool) -> Self {
        let mask_v1 = if v1 { BIT_MASK_32 } else { 0 };
        let mask_v2 = if v2 { BIT_MASK_32 } else { 0 };
        let mask_v3 = if v3 { BIT_MASK_32 } else { 0 };
        let mask_v4 = if v4 { BIT_MASK_32 } else { 0 };

        Self(i32x4(mask_v1, mask_v2, mask_v3, mask_v4))
    }

    pub fn splat(value: bool) -> Self {
        let mask = if value { BIT_MASK_32 } else { 0 };

        Self(i32x4_splat(mask))
    }

    pub fn extract_lanes(&self) -> (bool, bool, bool, bool) {
        (
            (i32x4_extract_lane::<0>(self.0) & BIT_MASK_32) != 0,
            (i32x4_extract_lane::<1>(self.0) & BIT_MASK_32) != 0,
            (i32x4_extract_lane::<2>(self.0) & BIT_MASK_32) != 0,
            (i32x4_extract_lane::<3>(self.0) & BIT_MASK_32) != 0,
        )
    }

    pub fn extract_lane(&self, index: usize) -> bool {
        match index {
            0 => (i32x4_extract_lane::<0>(self.0) & BIT_MASK_32) != 0,
            1 => (i32x4_extract_lane::<1>(self.0) & BIT_MASK_32) != 0,
            2 => (i32x4_extract_lane::<2>(self.0) & BIT_MASK_32) != 0,
            3 => (i32x4_extract_lane::<3>(self.0) & BIT_MASK_32) != 0,
            _ => panic!("Index out of bounds for Bx4"),
        }
    }

    pub fn set_lane(&mut self, index: usize, value: bool) {
        let mask_value = if value { BIT_MASK_32 } else { 0 };
        self.0 = match index {
            0 => i32x4_replace_lane::<0>(self.0, mask_value),
            1 => i32x4_replace_lane::<1>(self.0, mask_value),
            2 => i32x4_replace_lane::<2>(self.0, mask_value),
            3 => i32x4_replace_lane::<3>(self.0, mask_value),
            _ => panic!("Index out of bounds for Bx4"),
        };
    }

    pub fn to_bitmask(self) -> u8 {
        i32x4_bitmask(self.0)
    }

    pub(crate) fn from_v128(data: v128) -> Self {
        Self(data)
    }

    pub(crate) fn to_v128(self) -> v128 {
        self.0
    }
}

impl_default!(Bx4, bool);

impl Clone for Bx4 {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for Bx4 {}

impl BitAnd for Bx4 {
    type Output = Self;

    fn bitand(self, other: Self) -> Self::Output {
        Self(v128_and(self.0, other.0))
    }
}

impl BitAndAssign for Bx4 {
    fn bitand_assign(&mut self, other: Self) {
        self.0 = v128_and(self.0, other.0);
    }
}

impl BitAnd<bool> for Bx4 {
    type Output = Self;

    fn bitand(self, other: bool) -> Self::Output {
        let other_val = if other { BIT_MASK_32 } else { 0 };

        Self(v128_and(
            self.0,
            i32x4(other_val, other_val, other_val, other_val),
        ))
    }
}

impl BitAnd<Bx4> for bool {
    type Output = Bx4;

    fn bitand(self, other: Bx4) -> Self::Output {
        let self_val = if self { BIT_MASK_32 } else { 0 };

        Bx4(v128_and(
            i32x4(self_val, self_val, self_val, self_val),
            other.0,
        ))
    }
}

impl BitOr for Bx4 {
    type Output = Self;

    fn bitor(self, other: Self) -> Self::Output {
        Self(v128_or(self.0, other.0))
    }
}

impl BitOrAssign for Bx4 {
    fn bitor_assign(&mut self, other: Self) {
        self.0 = v128_or(self.0, other.0);
    }
}

impl BitOr<bool> for Bx4 {
    type Output = Self;

    fn bitor(self, other: bool) -> Self::Output {
        let other_val = if other { BIT_MASK_32 } else { 0 };
        Self(v128_or(
            self.0,
            i32x4(other_val, other_val, other_val, other_val),
        ))
    }
}

impl BitOr<Bx4> for bool {
    type Output = Bx4;

    fn bitor(self, other: Bx4) -> Self::Output {
        let self_val = if self { BIT_MASK_32 } else { 0 };
        Bx4(v128_or(
            i32x4(self_val, self_val, self_val, self_val),
            other.0,
        ))
    }
}

impl BitXor for Bx4 {
    type Output = Self;

    fn bitxor(self, other: Self) -> Self::Output {
        Self(v128_xor(self.0, other.0))
    }
}

impl BitXorAssign for Bx4 {
    fn bitxor_assign(&mut self, other: Self) {
        self.0 = v128_xor(self.0, other.0);
    }
}

impl BitXor<bool> for Bx4 {
    type Output = Self;

    fn bitxor(self, other: bool) -> Self::Output {
        let other_val = if other { BIT_MASK_32 } else { 0 };
        Self(v128_xor(
            self.0,
            i32x4(other_val, other_val, other_val, other_val),
        ))
    }
}

impl BitXor<Bx4> for bool {
    type Output = Bx4;

    fn bitxor(self, other: Bx4) -> Self::Output {
        let self_val = if self { BIT_MASK_32 } else { 0 };
        Bx4(v128_xor(
            i32x4(self_val, self_val, self_val, self_val),
            other.0,
        ))
    }
}

impl Not for Bx4 {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(v128_not(self.0))
    }
}

impl Debug for Bx4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (v1, v2, v3, v4) = self.extract_lanes();

        write!(f, "Bx4({}, {}, {}, {})", v1, v2, v3, v4)
    }
}
