.PHONY: all tests clippy

all:
	cargo build --target wasm32-unknown-unknown --release

tests:
	wasm-pack test --node

clippy:
	cargo clippy --target wasm32-unknown-unknown