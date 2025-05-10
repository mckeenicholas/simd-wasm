use wasm_bindgen_test::*;
use wasm_simd::Bx4;

fn assert_bx4_eq(a: Bx4, b: Bx4, msg: &str) {
    let lanes_a = a.extract_lanes();
    let lanes_b = b.extract_lanes();
    assert_eq!(lanes_a, lanes_b, "{}", msg);
}

#[wasm_bindgen_test]
fn test_new_and_extract_lanes() {
    let v = Bx4::new(true, false, true, false);
    assert_eq!(
        v.extract_lanes(),
        (true, false, true, false),
        "New and extract_lanes mismatch"
    );

    let v_all_true = Bx4::new(true, true, true, true);
    assert_eq!(
        v_all_true.extract_lanes(),
        (true, true, true, true),
        "New and extract_lanes (all true) mismatch"
    );

    let v_all_false = Bx4::new(false, false, false, false);
    assert_eq!(
        v_all_false.extract_lanes(),
        (false, false, false, false),
        "New and extract_lanes (all false) mismatch"
    );
}

#[wasm_bindgen_test]
fn test_splat() {
    let v_true = Bx4::splat(true);
    assert_eq!(
        v_true.extract_lanes(),
        (true, true, true, true),
        "Splat(true) mismatch"
    );

    let v_false = Bx4::splat(false);
    assert_eq!(
        v_false.extract_lanes(),
        (false, false, false, false),
        "Splat(false) mismatch"
    );
}

#[wasm_bindgen_test]
fn test_extract_lane() {
    let v = Bx4::new(true, false, true, false);
    assert_eq!(v.extract_lane(0), true, "Extract_lane(0) failed");
    assert_eq!(v.extract_lane(1), false, "Extract_lane(1) failed");
    assert_eq!(v.extract_lane(2), true, "Extract_lane(2) failed");
    assert_eq!(v.extract_lane(3), false, "Extract_lane(3) failed");
}

#[wasm_bindgen_test]
#[should_panic(expected = "Index out of bounds for Bx4")]
fn test_extract_lane_panic() {
    let v = Bx4::splat(false);
    v.extract_lane(4); // Index out of bounds
}

#[wasm_bindgen_test]
fn test_set_lane() {
    let mut v = Bx4::new(false, false, false, false);
    v.set_lane(0, true);
    assert_eq!(
        v.extract_lanes(),
        (true, false, false, false),
        "Set_lane(0, true) failed"
    );
    v.set_lane(2, true);
    assert_eq!(
        v.extract_lanes(),
        (true, false, true, false),
        "Set_lane(2, true) failed"
    );
    v.set_lane(0, false);
    assert_eq!(
        v.extract_lanes(),
        (false, false, true, false),
        "Set_lane(0, false) failed"
    );
}

#[wasm_bindgen_test]
#[should_panic(expected = "Index out of bounds for Bx4")]
fn test_set_lane_panic() {
    let mut v = Bx4::splat(true);
    v.set_lane(4, false); // Index out of bounds
}

#[wasm_bindgen_test]
fn test_to_bitmask() {
    assert_eq!(
        Bx4::new(false, false, false, false).to_bitmask(),
        0b0000,
        "Bitmask 0000 failed"
    );
    assert_eq!(
        Bx4::new(true, false, false, false).to_bitmask(),
        0b0001,
        "Bitmask 0001 failed"
    );
    assert_eq!(
        Bx4::new(false, true, false, false).to_bitmask(),
        0b0010,
        "Bitmask 0010 failed"
    );
    assert_eq!(
        Bx4::new(true, true, false, false).to_bitmask(),
        0b0011,
        "Bitmask 0011 failed"
    );
    assert_eq!(
        Bx4::new(false, false, true, false).to_bitmask(),
        0b0100,
        "Bitmask 0100 failed"
    );
    assert_eq!(
        Bx4::new(true, false, true, false).to_bitmask(),
        0b0101,
        "Bitmask 0101 failed"
    );
    assert_eq!(
        Bx4::new(false, true, true, false).to_bitmask(),
        0b0110,
        "Bitmask 0110 failed"
    );
    assert_eq!(
        Bx4::new(true, true, true, false).to_bitmask(),
        0b0111,
        "Bitmask 0111 failed"
    );
    assert_eq!(
        Bx4::new(false, false, false, true).to_bitmask(),
        0b1000,
        "Bitmask 1000 failed"
    );
    assert_eq!(
        Bx4::new(true, false, false, true).to_bitmask(),
        0b1001,
        "Bitmask 1001 failed"
    );
    assert_eq!(
        Bx4::new(false, true, false, true).to_bitmask(),
        0b1010,
        "Bitmask 1010 failed"
    );
    assert_eq!(
        Bx4::new(true, true, false, true).to_bitmask(),
        0b1011,
        "Bitmask 1011 failed"
    );
    assert_eq!(
        Bx4::new(false, false, true, true).to_bitmask(),
        0b1100,
        "Bitmask 1100 failed"
    );
    assert_eq!(
        Bx4::new(true, false, true, true).to_bitmask(),
        0b1101,
        "Bitmask 1101 failed"
    );
    assert_eq!(
        Bx4::new(false, true, true, true).to_bitmask(),
        0b1110,
        "Bitmask 1110 failed"
    );
    assert_eq!(
        Bx4::new(true, true, true, true).to_bitmask(),
        0b1111,
        "Bitmask 1111 failed"
    );
}

#[wasm_bindgen_test]
fn test_default() {
    let v_default = Bx4::default();
    assert_eq!(
        v_default.extract_lanes(),
        (false, false, false, false),
        "Default failed"
    );
}

#[wasm_bindgen_test]
fn test_clone_and_copy() {
    let v1 = Bx4::new(true, false, true, false);
    let v2 = v1.clone();
    assert_bx4_eq(v1, v2, "Clone failed");

    let mut v3 = Bx4::new(false, true, false, true);
    let v4 = v3; // Test Copy
    v3.set_lane(0, true); // Modify original after copy

    assert_bx4_eq(
        v4,
        Bx4::new(false, true, false, true),
        "Copy was not a true copy (v4 changed)",
    );
    assert_bx4_eq(
        v3,
        Bx4::new(true, true, false, true),
        "Original (v3) not modified as expected",
    );
}

#[wasm_bindgen_test]
fn test_bit_and() {
    let a = Bx4::new(true, true, false, false);
    let b = Bx4::new(true, false, true, false);
    assert_bx4_eq(
        a & b,
        Bx4::new(true, false, false, false),
        "BitAnd vector failed",
    );
    assert_bx4_eq(
        a & true,
        Bx4::new(true, true, false, false),
        "BitAnd scalar (true) failed",
    );
    assert_bx4_eq(
        a & false,
        Bx4::new(false, false, false, false),
        "BitAnd scalar (false) failed",
    );
    assert_bx4_eq(
        true & b,
        Bx4::new(true, false, true, false),
        "BitAnd scalar (true) LHS failed",
    );
    assert_bx4_eq(
        false & b,
        Bx4::new(false, false, false, false),
        "BitAnd scalar (false) LHS failed",
    );
}

#[wasm_bindgen_test]
fn test_bit_and_assign() {
    let mut a = Bx4::new(true, true, false, false);
    let b = Bx4::new(true, false, true, false);
    a &= b;
    assert_bx4_eq(
        a,
        Bx4::new(true, false, false, false),
        "BitAndAssign vector failed",
    );

    let mut c = Bx4::new(true, true, false, false);
    c &= true;
    assert_bx4_eq(
        c,
        Bx4::new(true, true, false, false),
        "BitAndAssign scalar (true) failed",
    );

    let mut d = Bx4::new(true, true, false, false);
    d &= false;
    assert_bx4_eq(
        d,
        Bx4::new(false, false, false, false),
        "BitAndAssign scalar (false) failed",
    );
}

#[wasm_bindgen_test]
fn test_bit_or() {
    let a = Bx4::new(true, true, false, false);
    let b = Bx4::new(true, false, true, false);
    assert_bx4_eq(
        a | b,
        Bx4::new(true, true, true, false),
        "BitOr vector failed",
    );
    assert_bx4_eq(
        a | true,
        Bx4::new(true, true, true, true),
        "BitOr scalar (true) failed",
    );
    assert_bx4_eq(
        a | false,
        Bx4::new(true, true, false, false),
        "BitOr scalar (false) failed",
    );
    assert_bx4_eq(
        true | b,
        Bx4::new(true, true, true, true),
        "BitOr scalar (true) LHS failed",
    );
    assert_bx4_eq(
        false | b,
        Bx4::new(true, false, true, false),
        "BitOr scalar (false) LHS failed",
    );
}

#[wasm_bindgen_test]
fn test_bit_or_assign() {
    let mut a = Bx4::new(true, true, false, false);
    let b = Bx4::new(true, false, true, false);
    a |= b;
    assert_bx4_eq(
        a,
        Bx4::new(true, true, true, false),
        "BitOrAssign vector failed",
    );

    let mut c = Bx4::new(true, true, false, false);
    c |= true;
    assert_bx4_eq(
        c,
        Bx4::new(true, true, true, true),
        "BitOrAssign scalar (true) failed",
    );

    let mut d = Bx4::new(true, true, false, false);
    d |= false; // Should not change
    assert_bx4_eq(
        d,
        Bx4::new(true, true, false, false),
        "BitOrAssign scalar (false) failed",
    );
}

#[wasm_bindgen_test]
fn test_bit_xor() {
    let a = Bx4::new(true, true, false, false);
    let b = Bx4::new(true, false, true, false);
    assert_bx4_eq(
        a ^ b,
        Bx4::new(false, true, true, false),
        "BitXor vector failed",
    );
    assert_bx4_eq(
        a ^ true,
        Bx4::new(false, false, true, true),
        "BitXor scalar (true) failed",
    );
    assert_bx4_eq(
        a ^ false,
        Bx4::new(true, true, false, false),
        "BitXor scalar (false) failed",
    );
    assert_bx4_eq(
        true ^ b,
        Bx4::new(false, true, false, true),
        "BitXor scalar (true) LHS failed",
    );
    assert_bx4_eq(
        false ^ b,
        Bx4::new(true, false, true, false),
        "BitXor scalar (false) LHS failed",
    );
}

#[wasm_bindgen_test]
fn test_bit_xor_assign() {
    let mut a = Bx4::new(true, true, false, false);
    let b = Bx4::new(true, false, true, false);
    a ^= b;
    assert_bx4_eq(
        a,
        Bx4::new(false, true, true, false),
        "BitXorAssign vector failed",
    );

    let mut c = Bx4::new(true, true, false, false);
    c ^= true;
    assert_bx4_eq(
        c,
        Bx4::new(false, false, true, true),
        "BitXorAssign scalar (true) failed",
    );

    let mut d = Bx4::new(true, true, false, false);
    d ^= false; // Should not change
    assert_bx4_eq(
        d,
        Bx4::new(true, true, false, false),
        "BitXorAssign scalar (false) failed",
    );
}

#[wasm_bindgen_test]
fn test_not() {
    let a = Bx4::new(true, false, true, false);
    assert_bx4_eq(!a, Bx4::new(false, true, false, true), "Not failed");

    let b = Bx4::splat(true);
    assert_bx4_eq(!b, Bx4::splat(false), "Not all true failed");

    let c = Bx4::splat(false);
    assert_bx4_eq(!c, Bx4::splat(true), "Not all false failed");
}

#[wasm_bindgen_test]
fn test_debug_format() {
    let v = Bx4::new(true, false, true, false);
    let formatted = format!("{:?}", v);
    assert_eq!(
        formatted, "Bx4(true, false, true, false)",
        "Debug format incorrect"
    );

    let v_all_true = Bx4::splat(true);
    let formatted_true = format!("{:?}", v_all_true);
    assert_eq!(
        formatted_true, "Bx4(true, true, true, true)",
        "Debug format all true incorrect"
    );
}
