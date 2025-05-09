use std::i32;
use wasm_bindgen_test::*;
use wasm_simd::{Bx4, F32x4, I32x4};

fn assert_i32x4_eq(a: I32x4, b: I32x4, msg: &str) {
    let lanes_a = a.extract_lanes();
    let lanes_b = b.extract_lanes();
    assert_eq!(lanes_a, lanes_b, "{}", msg);
}

fn assert_bx4_eq(a: Bx4, b: Bx4, msg: &str) {
    let lanes_a = a.extract_lanes();
    let lanes_b = b.extract_lanes();
    assert_eq!(lanes_a, lanes_b, "{}", msg);
}

#[wasm_bindgen_test]
fn test_new_and_extract_lanes() {
    let v = I32x4::new(1, 2, 3, 4);
    assert_eq!(
        v.extract_lanes(),
        (1, 2, 3, 4),
        "New and extract_lanes mismatch"
    );
}

#[wasm_bindgen_test]
fn test_splat() {
    let v = I32x4::splat(7);
    assert_eq!(v.extract_lanes(), (7, 7, 7, 7), "Splat mismatch");
}

#[wasm_bindgen_test]
fn test_new_from_fn() {
    let v = I32x4::new_from_fn(|i| (i as i32 + 1) * 10);
    assert_eq!(v.extract_lanes(), (10, 20, 30, 40), "New_from_fn mismatch");
}

#[wasm_bindgen_test]
fn test_extract_lane() {
    let v = I32x4::new(10, 20, 30, 40);
    assert_eq!(v.extract_lane(0), 10, "Extract_lane(0) failed");
    assert_eq!(v.extract_lane(1), 20, "Extract_lane(1) failed");
    assert_eq!(v.extract_lane(2), 30, "Extract_lane(2) failed");
    assert_eq!(v.extract_lane(3), 40, "Extract_lane(3) failed");
}

#[wasm_bindgen_test]
#[should_panic(expected = "Index out of bounds")]
fn test_extract_lane_panic() {
    let v = I32x4::splat(0);
    v.extract_lane(4);
}

#[wasm_bindgen_test]
fn test_set_lane() {
    let mut v = I32x4::new(1, 2, 3, 4);
    v.set_lane(0, 11);
    v.set_lane(2, 33);
    assert_eq!(v.extract_lanes(), (11, 2, 33, 4), "Set_lane failed");
}

#[wasm_bindgen_test]
#[should_panic(expected = "Index out of bounds")]
fn test_set_lane_panic() {
    let mut v = I32x4::splat(0);
    v.set_lane(4, 100);
}

#[wasm_bindgen_test]
fn test_apply() {
    let v = I32x4::new(1, -2, 3, -4);
    let applied = v.apply(|x| x * x);
    assert_i32x4_eq(applied, I32x4::new(1, 4, 9, 16), "Apply failed");
}

#[wasm_bindgen_test]
fn test_fold() {
    let v = I32x4::new(1, 2, 3, 4);
    let sum = v.fold(|acc, x| acc + x, 10);
    assert_eq!(sum, 10 + 1 + 2 + 3 + 4, "Fold sum failed");
    let product = v.fold(|acc, x| acc * x, 2);
    assert_eq!(product, 2 * 1 * 2 * 3 * 4, "Fold product failed");
}

#[wasm_bindgen_test]
fn test_reduce_add() {
    let v = I32x4::new(1, 2, 3, 4);
    assert_eq!(v.reduce_add(), 10, "Reduce_add failed");
    let v_neg = I32x4::new(-1, -2, -3, -4);
    assert_eq!(v_neg.reduce_add(), -10, "Reduce_add with negatives failed");
}

#[wasm_bindgen_test]
fn test_reduce_mul() {
    let v = I32x4::new(1, 2, 3, 4);
    assert_eq!(v.reduce_mul(), 24, "Reduce_mul failed");
    let v_zero = I32x4::new(1, 0, 3, 4);
    assert_eq!(v_zero.reduce_mul(), 0, "Reduce_mul with zero failed");
}

#[wasm_bindgen_test]
fn test_reduce_min() {
    let v = I32x4::new(5, 1, 9, 3);
    assert_eq!(v.reduce_min(), 1, "Reduce_min failed");
    let v_neg = I32x4::new(-5, -1, -9, -3);
    assert_eq!(v_neg.reduce_min(), -9, "Reduce_min with negatives failed");
}

#[wasm_bindgen_test]
fn test_reduce_max() {
    let v = I32x4::new(5, 1, 9, 3);
    assert_eq!(v.reduce_max(), 9, "Reduce_max failed");
    let v_neg = I32x4::new(-5, -1, -9, -3);
    assert_eq!(v_neg.reduce_max(), -1, "Reduce_max with negatives failed");
}

#[wasm_bindgen_test]
fn test_if_else() {
    let a = I32x4::new(1, 2, 3, 4);
    let b = I32x4::new(10, 20, 30, 40);
    let mask_ttff = Bx4::new(true, true, false, false);
    let result = a.if_else(&b, &mask_ttff);
    assert_i32x4_eq(result, I32x4::new(1, 2, 30, 40), "If_else ttff failed");

    let mask_ftft = Bx4::new(false, true, false, true);
    let result2 = a.if_else(&b, &mask_ftft);
    assert_i32x4_eq(result2, I32x4::new(10, 2, 30, 4), "If_else ftft failed");
}

#[wasm_bindgen_test]
fn test_all_nonzero() {
    let v1 = I32x4::new(1, 2, 3, 4);
    assert!(v1.all_nonzero(), "All_nonzero for (1,2,3,4) failed");
    let v2 = I32x4::new(1, 0, 3, 4);
    assert!(!v2.all_nonzero(), "All_nonzero for (1,0,3,4) failed");
    let v3 = I32x4::splat(0);
    assert!(!v3.all_nonzero(), "All_nonzero for (0,0,0,0) failed");
    let v4 = I32x4::new(-1, -2, -3, -4);
    assert!(v4.all_nonzero(), "All_nonzero for (-1,-2,-3,-4) failed");
}

#[wasm_bindgen_test]
fn test_shuffle() {
    let a = I32x4::new(1, 2, 3, 4);
    let b = I32x4::new(10, 20, 30, 40);
    // Select a0, b1, a2, b3
    let shuffled = a.shuffle::<0, 5, 2, 7>(&b); // Lanes 0-3 from a, 4-7 from b
    assert_i32x4_eq(shuffled, I32x4::new(1, 20, 3, 40), "Shuffle failed");

    // Reverse a
    let reversed_a = a.shuffle::<3, 2, 1, 0>(&a);
    assert_i32x4_eq(reversed_a, I32x4::new(4, 3, 2, 1), "Shuffle reverse failed");
}

#[wasm_bindgen_test]
fn test_comparisons() {
    let a = I32x4::new(1, 2, 3, 4);
    let b = I32x4::new(1, 0, 5, 4);

    assert_bx4_eq(
        a.eq(&b),
        Bx4::new(true, false, false, true),
        "eq vector failed",
    );
    assert_bx4_eq(
        a.s_eq(1),
        Bx4::new(true, false, false, false),
        "s_eq scalar failed",
    );

    assert_bx4_eq(
        a.ne(&b),
        Bx4::new(false, true, true, false),
        "ne vector failed",
    );
    assert_bx4_eq(
        a.s_ne(1),
        Bx4::new(false, true, true, true),
        "s_ne scalar failed",
    );

    assert_bx4_eq(
        a.lt(&b),
        Bx4::new(false, false, true, false),
        "lt vector failed",
    );
    assert_bx4_eq(
        a.s_lt(3),
        Bx4::new(true, true, false, false),
        "s_lt scalar failed",
    );

    assert_bx4_eq(
        a.le(&b),
        Bx4::new(true, false, true, true),
        "le vector failed",
    );
    assert_bx4_eq(
        a.s_le(3),
        Bx4::new(true, true, true, false),
        "s_le scalar failed",
    );

    assert_bx4_eq(
        a.gt(&b),
        Bx4::new(false, true, false, false),
        "gt vector failed",
    );
    assert_bx4_eq(
        a.s_gt(3),
        Bx4::new(false, false, false, true),
        "s_gt scalar failed",
    );

    assert_bx4_eq(
        a.ge(&b),
        Bx4::new(true, true, false, true),
        "ge vector failed",
    );
    assert_bx4_eq(
        a.s_ge(3),
        Bx4::new(false, false, true, true),
        "s_ge scalar failed",
    );
}

#[wasm_bindgen_test]
fn test_min_max_binary_ops() {
    let a = I32x4::new(1, 5, 2, 8);
    let b = I32x4::new(3, 2, 7, 8);

    assert_i32x4_eq(a.min(&b), I32x4::new(1, 2, 2, 8), "min vector failed");
    assert_i32x4_eq(a.s_min(4), I32x4::new(1, 4, 2, 4), "s_min scalar failed");

    assert_i32x4_eq(a.max(&b), I32x4::new(3, 5, 7, 8), "max vector failed");
    assert_i32x4_eq(a.s_max(4), I32x4::new(4, 5, 4, 8), "s_max scalar failed");
}

#[wasm_bindgen_test]
fn test_abs() {
    let v = I32x4::new(1, -2, 0, -i32::MAX);
    let v_abs = v.abs();
    assert_i32x4_eq(v_abs, I32x4::new(1, 2, 0, i32::MAX), "abs failed");

    let v2 = I32x4::new(-5, -i32::MAX, i32::MAX, -0);
    let v2_abs = v2.abs();
    assert_i32x4_eq(
        v2_abs,
        I32x4::new(5, i32::MAX, i32::MAX, 0),
        "abs detailed failed",
    );
}

#[wasm_bindgen_test]
fn test_default() {
    let v1 = I32x4::default();

    assert_eq!(v1.extract_lanes(), (0, 0, 0, 0), "default failed")
}

#[wasm_bindgen_test]
fn test_clone() {
    let v1 = I32x4::new(1, 2, 3, 4);
    let v2 = v1.clone();
    assert_i32x4_eq(v1.clone(), v2, "Clone failed, values mismatch");
    // Modify original to ensure it's a deep clone (though v128 is Copy)
    // This test is more about the Clone trait being correctly implemented
    let mut v3 = v1.clone();
    v3.set_lane(0, 100);
    assert_i32x4_eq(
        v1,
        I32x4::new(1, 2, 3, 4),
        "Original modified after clone's modification",
    );
    assert_i32x4_eq(
        v3,
        I32x4::new(100, 2, 3, 4),
        "Cloned value not modified correctly",
    );
}

#[wasm_bindgen_test]
fn test_into_array() {
    let v = I32x4::new(10, 20, 30, 40);
    let arr: [i32; 4] = v.into();
    assert_eq!(arr, [10, 20, 30, 40], "Into<[i32; 4]> failed");
}

#[wasm_bindgen_test]
fn test_into_vec() {
    let v = I32x4::new(11, 22, 33, 44);
    let vec: Vec<i32> = v.into();
    assert_eq!(vec, vec![11, 22, 33, 44], "Into<Vec<i32>> failed");
}

#[wasm_bindgen_test]
fn test_from_f32x4() {
    let fv = F32x4::new(1.1, -2.9, 3.5, -4.0001);
    let iv = I32x4::from(fv); // i32x4_trunc_sat_f32x4
    assert_i32x4_eq(
        iv,
        I32x4::new(1, -2, 3, -4),
        "From<F32x4> basic truncation failed",
    );

    let fv_large = F32x4::new(f32::MAX, f32::MIN, 10.0, -10.0);
    let iv_sat = I32x4::from(fv_large);
    assert_i32x4_eq(
        iv_sat,
        I32x4::new(i32::MAX, i32::MIN, 10, -10),
        "From<F32x4> saturation failed",
    );
}

#[wasm_bindgen_test]
fn test_from_array() {
    let arr = [5, 6, 7, 8];
    let v = I32x4::from(arr);
    assert_i32x4_eq(v, I32x4::new(5, 6, 7, 8), "From<[i32; 4]> failed");
}

#[wasm_bindgen_test]
fn test_debug_format() {
    let v = I32x4::new(1, -2, 3, -4);
    let formatted = format!("{:?}", v);
    assert_eq!(formatted, "I32x4(1, -2, 3, -4)", "Debug format incorrect");
}

#[wasm_bindgen_test]
fn test_neg() {
    let v = I32x4::new(1, -2, 0, i32::MIN);
    let neg_v = -v;
    // For WASM i32x4_neg, neg(i32::MIN) = i32::MIN.
    assert_i32x4_eq(neg_v, I32x4::new(-1, 2, 0, i32::MIN), "Negation failed");
}

#[wasm_bindgen_test]
fn test_add() {
    let a = I32x4::new(1, 2, 3, 100);
    let b = I32x4::new(10, 20, 30, -50);
    assert_i32x4_eq(
        a.clone() + b,
        I32x4::new(11, 22, 33, 50),
        "Add vector + vector failed",
    );
    assert_i32x4_eq(
        a.clone() + 5,
        I32x4::new(6, 7, 8, 105),
        "Add vector + scalar failed",
    );
    assert_i32x4_eq(
        5 + a,
        I32x4::new(6, 7, 8, 105),
        "Add scalar + vector failed",
    );
}

#[wasm_bindgen_test]
fn test_sub() {
    let a = I32x4::new(10, 20, 30, 100);
    let b = I32x4::new(1, 2, 5, -50);
    assert_i32x4_eq(
        a.clone() - b,
        I32x4::new(9, 18, 25, 150),
        "Sub vector - vector failed",
    );
    assert_i32x4_eq(
        a.clone() - 5,
        I32x4::new(5, 15, 25, 95),
        "Sub vector - scalar failed",
    );
    assert_i32x4_eq(
        100 - a,
        I32x4::new(90, 80, 70, 0),
        "Sub scalar - vector failed",
    );
}

#[wasm_bindgen_test]
fn test_mul() {
    let a = I32x4::new(1, 2, -3, 10);
    let b = I32x4::new(5, -4, 2, 100);
    assert_i32x4_eq(
        a.clone() * b,
        I32x4::new(5, -8, -6, 1000),
        "Mul vector * vector failed",
    );
    assert_i32x4_eq(
        a.clone() * 3,
        I32x4::new(3, 6, -9, 30),
        "Mul vector * scalar failed",
    );
    assert_i32x4_eq(
        3 * a,
        I32x4::new(3, 6, -9, 30),
        "Mul scalar * vector failed",
    );
}

#[wasm_bindgen_test]
fn test_div() {
    let a = I32x4::new(10, 21, -9, 100);
    let b = I32x4::new(2, 7, 3, -10);
    assert_i32x4_eq(
        a.clone() / b,
        I32x4::new(5, 3, -3, -10),
        "Div vector / vector failed",
    );
    assert_i32x4_eq(
        a / 2,
        I32x4::new(5, 10, -4, 50),
        "Div vector / scalar failed",
    ); // Note integer division
    assert_i32x4_eq(
        100 / I32x4::new(10, -5, 4, 50),
        I32x4::new(10, -20, 25, 2),
        "Div scalar / vector failed",
    );
}

#[wasm_bindgen_test]
#[should_panic]
fn test_div_by_zero_vector() {
    let a = I32x4::new(10, 20, 30, 40);
    let b = I32x4::new(1, 0, 2, 3); // Division by zero in lane 1
    let _ = a / b;
}

#[wasm_bindgen_test]
#[should_panic]
fn test_div_by_zero_scalar() {
    let a = I32x4::new(10, 20, 30, 40);
    let _ = a / 0;
}

#[wasm_bindgen_test]
fn test_shl() {
    let a = I32x4::new(1, 2, 3, 4);
    assert_i32x4_eq(a.clone() << 1u32, I32x4::new(2, 4, 6, 8), "Shl by 1 failed");
    assert_i32x4_eq(a << 3u32, I32x4::new(8, 16, 24, 32), "Shl by 3 failed");
    assert_i32x4_eq(
        I32x4::new(i32::MAX, -1, 1, 0) << 1u32,
        I32x4::new(-2, -2, 2, 0),
        "Shl with overflow/sign change",
    );
}

#[wasm_bindgen_test]
fn test_shr() {
    // This will be arithmetic shift right (SAR) due to i32
    let a = I32x4::new(8, -8, 7, -7);
    assert_i32x4_eq(
        a.clone() >> 1u32,
        I32x4::new(4, -4, 3, -4),
        "Shr by 1 failed",
    ); // -7/2 = -3.5 -> -4
    assert_i32x4_eq(a >> 2u32, I32x4::new(2, -2, 1, -2), "Shr by 2 failed"); // -7 >> 2 = -2
    let b = I32x4::new(i32::MIN, i32::MAX, 0, -1);
    assert_i32x4_eq(
        b >> 1u32,
        I32x4::new(i32::MIN / 2, i32::MAX / 2, 0, -1),
        "Shr edge cases",
    );
}

#[wasm_bindgen_test]
fn test_add_assign() {
    let mut a = I32x4::new(1, 2, 3, 10);
    a += I32x4::new(10, 20, 30, -5);
    assert_i32x4_eq(a, I32x4::new(11, 22, 33, 5), "AddAssign vector failed");

    let mut b = I32x4::new(1, 2, 3, 10);
    b += 5;
    assert_i32x4_eq(b, I32x4::new(6, 7, 8, 15), "AddAssign scalar failed");
}

#[wasm_bindgen_test]
fn test_sub_assign() {
    let mut a = I32x4::new(10, 20, 30, 10);
    a -= I32x4::new(1, 2, 5, -5);
    assert_i32x4_eq(a, I32x4::new(9, 18, 25, 15), "SubAssign vector failed");

    let mut b = I32x4::new(10, 20, 30, 10);
    b -= 5;
    assert_i32x4_eq(b, I32x4::new(5, 15, 25, 5), "SubAssign scalar failed");
}

#[wasm_bindgen_test]
fn test_mul_assign() {
    let mut a = I32x4::new(1, 2, -3, 10);
    a *= I32x4::new(5, -4, 2, 2);
    assert_i32x4_eq(a, I32x4::new(5, -8, -6, 20), "MulAssign vector failed");

    let mut b = I32x4::new(1, 2, -3, 10);
    b *= 3;
    assert_i32x4_eq(b, I32x4::new(3, 6, -9, 30), "MulAssign scalar failed");
}

#[wasm_bindgen_test]
fn test_div_assign() {
    let mut a = I32x4::new(10, 21, -9, 100);
    a /= I32x4::new(2, 7, 3, -10);
    assert_i32x4_eq(a, I32x4::new(5, 3, -3, -10), "DivAssign vector failed");

    let mut b = I32x4::new(10, 21, -9, 100);
    b /= 2;
    assert_i32x4_eq(b, I32x4::new(5, 10, -4, 50), "DivAssign scalar failed");
}

#[wasm_bindgen_test]
#[should_panic]
fn test_div_assign_by_zero_vector() {
    let mut a = I32x4::new(10, 20, 30, 40);
    a /= I32x4::new(1, 0, 2, 3);
}

#[wasm_bindgen_test]
#[should_panic]
fn test_div_assign_by_zero_scalar() {
    let mut a = I32x4::new(10, 20, 30, 40);
    a /= 0;
}

#[wasm_bindgen_test]
fn test_shl_assign() {
    let mut a = I32x4::new(1, 2, 3, 4);
    a <<= 2u32;
    assert_i32x4_eq(a, I32x4::new(4, 8, 12, 16), "ShlAssign failed");
}

#[wasm_bindgen_test]
fn test_shr_assign() {
    let mut a = I32x4::new(8, -8, 7, -7);
    a >>= 1u32;
    assert_i32x4_eq(a, I32x4::new(4, -4, 3, -4), "ShrAssign failed");
}
