pub mod bx4;
pub mod f32x4;
pub mod i32x4;
pub mod u32x4;

#[macro_export]
macro_rules! impl_vec_binary_op {
    ($name:ident, $scalar_name:ident, $fn:ident, $type:ty) => {
        #[inline]
        pub fn $name(self, other: Self) -> Self {
            Self(::core::arch::wasm32::$fn(self.0, other.0))
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
            Self(::core::arch::wasm32::$fn(self.0))
        }
    };
}

#[macro_export]
macro_rules! impl_vec_overload_op {
    ($vec_type:ty, $scalar_type:ty, $trait:ident, $fn:ident, $op_fn:ident) => {
        impl $trait for $vec_type {
            type Output = Self;
            #[inline]
            fn $fn(self, other: Self) -> Self::Output {
                Self($op_fn(self.0, other.0))
            }
        }

        impl $trait<$scalar_type> for $vec_type {
            type Output = Self;
            #[inline]
            fn $fn(self, other: $scalar_type) -> Self::Output {
                Self($op_fn(self.0, Self::splat(other).to_v128()))
            }
        }

        impl $trait<$vec_type> for $scalar_type {
            type Output = $vec_type;
            #[inline]

            fn $fn(self, other: $vec_type) -> Self::Output {
                <$vec_type>::from_v128($op_fn(<$vec_type>::splat(self).to_v128(), other.0))
            }
        }
    };
}

#[macro_export]
macro_rules! impl_vec_assign_op {
    ($vec_type:ty, $scalar_type: ty, $trait:ident, $fn:ident, $op:tt) => {
        impl $trait for $vec_type {
            #[inline]
            fn $fn(&mut self, other: Self) {
                (*self) = self.clone() $op other;
            }
        }

        impl $trait<$scalar_type> for $vec_type {
            #[inline]
            fn $fn(&mut self, other: $scalar_type) {
                (*self) = self.clone() $op Self::splat(other);
            }
        }
    };
}

#[macro_export]
macro_rules! impl_vec_cmp {
    ($vec_fn:ident, $scalar_fn:ident, $cmp_fn:ident, $ret:ty) => {
        #[inline]
        pub fn $vec_fn(self, other: Self) -> $ret {
            let mask = ::core::arch::wasm32::$cmp_fn(self.0, other.0);
            <$ret>::from_v128(mask)
        }

        #[inline]
        pub fn $scalar_fn(self, other: i32) -> $ret {
            let mask = ::core::arch::wasm32::$cmp_fn(self.0, i32x4_splat(other));
            <$ret>::from_v128(mask)
        }
    };
}

#[macro_export]
macro_rules! impl_debug {
    ($struct_name:ident, ($($field_var:ident),+)) => {
        impl std::fmt::Debug for $struct_name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let ($($field_var),+) = self.extract_lanes();
                f.debug_tuple(stringify!($struct_name))
                    $(.field(&$field_var))+
                    .finish()
            }
        }
    };
}
