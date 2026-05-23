#!/bin/bash
REG=/Users/manhvu/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f

echo "=== Burn packages ==="
ls "$REG/" | grep burn

echo ""
echo "=== CubeCL lib.rs head ==="
head -100 "$REG/burn-cubecl-0.21.0/src/lib.rs" 2>/dev/null || echo "NOT FOUND"

echo ""
echo "=== burn-tensor mod.rs head ==="
head -100 "$REG/burn-0.21.0/src/tensor/mod.rs" 2>/dev/null || echo "NOT FOUND"

echo ""
echo "=== Data struct in burn-tensor ==="
grep -rn "pub struct Data" "$REG/burn-tensor-0.21.0/src/" 2>/dev/null | head -10

echo ""
echo "=== from_data in burn-tensor ==="
grep -rn "pub fn from_data\|pub fn from_data_sync" "$REG/burn-tensor-0.21.0/src/" 2>/dev/null | head -10

echo ""
echo "=== CubeCL struct ==="
grep -rn "pub struct CubeCL\|pub type CubeCL" "$REG/burn-cubecl-0.21.0/src/" 2>/dev/null | head -10

echo ""
echo "=== burn backend mod ==="
grep -rn "pub use\|pub mod" "$REG/burn-0.21.0/src/backend/mod.rs" 2>/dev/null | head -20

echo ""
echo "=== Dynamic dims ==="
grep -rn "Dynamic\|pub type Dynamic" "$REG/burn-tensor-0.21.0/src/" 2>/dev/null | head -10

echo ""
echo "=== Tensor creation methods ==="
grep -rn "pub fn from_data\|pub fn random\|pub fn zeros\|pub fn ones\|pub fn empty" "$REG/burn-tensor-0.21.0/src/tensor/api/" 2>/dev/null | head -20

echo ""
echo "=== burn lib.rs re-exports ==="
head -80 "$REG/burn-0.21.0/src/lib.rs" 2>/dev/null
