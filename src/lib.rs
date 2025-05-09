pub mod bx4;
pub mod f32x4;
pub mod i32x4;

#[macro_export]
macro_rules! impl_vec_op {
    ($name:ident, $scalar_name:ident, $fn:ident, $type:ty) => {
        #[inline]
        pub fn $name(self, other: Self) -> Self {
            Self {
                data: ::core::arch::wasm32::$fn(self.data, other.data),
            }
        }

        #[inline]
        pub fn $scalar_name(self, other: $type) -> Self {
            self.$name(Self::splat(other))
        }
    };
}

#[macro_export]
macro_rules! impl_vec_unary_op {
    ($name:ident, $fn:ident) => {
        #[inline]
        pub fn $name(self) -> Self {
            Self {
                data: ::core::arch::wasm32::$fn(self.data),
            }
        }
    };
}

#[macro_export]
macro_rules! impl_vec_cmp {
    ($vec_fn:ident, $scalar_fn:ident, $cmp_fn:ident, $ret:ty) => {
        #[inline]
        pub fn $vec_fn(self, other: Self) -> $ret {
            let mask = ::core::arch::wasm32::$cmp_fn(self.data, other.data);
            <$ret>::from_v128(mask)
        }

        #[inline]
        pub fn $scalar_fn(self, other: i32) -> $ret {
            let mask = ::core::arch::wasm32::$cmp_fn(self.data, i32x4_splat(other));
            <$ret>::from_v128(mask)
        }
    };
}
