use core::arch::wasm32::{
    i32x4, i32x4_bitmask, i32x4_extract_lane, v128, v128_and, v128_not, v128_or, v128_xor,
};
use std::fmt::Debug;
use std::ops::Not;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign};

const BIT_MASK_32: i32 = -1i32;

pub struct Bx4 {
    pub(crate) data: v128,
}

impl Bx4 {
    pub fn default() -> Self {
        Self {
            data: i32x4(0, 0, 0, 0),
        }
    }

    pub fn new(v1: bool, v2: bool, v3: bool, v4: bool) -> Self {
        let mask_v1 = if v1 { BIT_MASK_32 } else { 0 };
        let mask_v2 = if v2 { BIT_MASK_32 } else { 0 };
        let mask_v3 = if v3 { BIT_MASK_32 } else { 0 };
        let mask_v4 = if v4 { BIT_MASK_32 } else { 0 };

        Self {
            data: i32x4(mask_v1, mask_v2, mask_v3, mask_v4),
        }
    }

    #[inline]
    pub fn to_bitmask(self) -> u8 {
        i32x4_bitmask(self.data)
    }

    pub(crate) fn from_v128(data: v128) -> Self {
        Self { data }
    }
}

impl BitAnd for Bx4 {
    type Output = Self;

    fn bitand(self, other: Self) -> Self::Output {
        Self {
            data: v128_and(self.data, other.data),
        }
    }
}

impl BitAndAssign for Bx4 {
    fn bitand_assign(&mut self, other: Self) {
        self.data = v128_and(self.data, other.data);
    }
}

impl BitAnd<bool> for Bx4 {
    type Output = Self;

    fn bitand(self, other: bool) -> Self::Output {
        let other = if other { BIT_MASK_32 } else { 0 };

        Self {
            data: v128_and(self.data, i32x4(other, other, other, other)),
        }
    }
}

impl BitAnd<Bx4> for bool {
    type Output = Bx4;

    fn bitand(self, other: Bx4) -> Self::Output {
        let self_val = if self { BIT_MASK_32 } else { 0 };

        Bx4 {
            data: v128_and(i32x4(self_val, self_val, self_val, self_val), other.data),
        }
    }
}

impl BitOr for Bx4 {
    type Output = Self;

    fn bitor(self, other: Self) -> Self::Output {
        Self {
            data: v128_or(self.data, other.data),
        }
    }
}

impl BitOrAssign for Bx4 {
    fn bitor_assign(&mut self, other: Self) {
        self.data = v128_or(self.data, other.data);
    }
}

impl BitOr<bool> for Bx4 {
    type Output = Self;

    fn bitor(self, other: bool) -> Self::Output {
        let other_val = if other { BIT_MASK_32 } else { 0 };
        Self {
            data: v128_or(self.data, i32x4(other_val, other_val, other_val, other_val)),
        }
    }
}

impl BitOr<Bx4> for bool {
    type Output = Bx4;

    fn bitor(self, other: Bx4) -> Self::Output {
        let self_val = if self { BIT_MASK_32 } else { 0 };
        Bx4 {
            data: v128_or(i32x4(self_val, self_val, self_val, self_val), other.data),
        }
    }
}

impl BitXor for Bx4 {
    type Output = Self;

    fn bitxor(self, other: Self) -> Self::Output {
        Self {
            data: v128_xor(self.data, other.data),
        }
    }
}

impl BitXorAssign for Bx4 {
    fn bitxor_assign(&mut self, other: Self) {
        self.data = v128_xor(self.data, other.data);
    }
}

impl BitXor<bool> for Bx4 {
    type Output = Self;

    fn bitxor(self, other: bool) -> Self::Output {
        let other_val = if other { BIT_MASK_32 } else { 0 };
        Self {
            data: v128_xor(self.data, i32x4(other_val, other_val, other_val, other_val)),
        }
    }
}

impl BitXor<Bx4> for bool {
    type Output = Bx4;

    fn bitxor(self, other: Bx4) -> Self::Output {
        let self_val = if self { BIT_MASK_32 } else { 0 };
        Bx4 {
            data: v128_xor(i32x4(self_val, self_val, self_val, self_val), other.data),
        }
    }
}

impl Not for Bx4 {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self {
            data: v128_not(self.data),
        }
    }
}

impl Debug for Bx4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const MASK: i32 = BIT_MASK_32;
        let v1 = (i32x4_extract_lane::<0>(self.data) & MASK) != 0;
        let v2 = (i32x4_extract_lane::<1>(self.data) & MASK) != 0;
        let v3 = (i32x4_extract_lane::<2>(self.data) & MASK) != 0;
        let v4 = (i32x4_extract_lane::<3>(self.data) & MASK) != 0;

        write!(f, "Bx4({}, {}, {}, {})", v1, v2, v3, v4)
    }
}
