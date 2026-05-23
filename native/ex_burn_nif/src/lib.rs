//! # ExBurn NIF
//!
//! Rust NIF bridge between Elixir and the Burn deep learning framework.
//! Uses Burn's NdArray backend with autodiff support.

use rustler::{Atom, Env, ResourceArc, Term};
use std::panic::{RefUnwindSafe, UnwindSafe};

// ── Burn imports ────────────────────────────────────────────────────
use burn::tensor::Tensor;
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;

// ── Backend type ───────────────────────────────────────────────────
type B = Autodiff<NdArray>;

fn device() -> burn_ndarray::NdArrayDevice {
    burn_ndarray::NdArrayDevice::default()
}

// ── Tensor kind enum ──────────────────────────────────────────────
/// Enum wrapping concrete Burn tensor types so they can be stored in a
/// single registry.
#[derive(Clone)]
pub enum BurnTensor {
    F32x1(Tensor<B, 1>),
    F32x2(Tensor<B, 2>),
    F32x3(Tensor<B, 3>),
    F32x4(Tensor<B, 4>),
    I32x1(Tensor<B, 1, burn::tensor::Int>),
    I32x2(Tensor<B, 2, burn::tensor::Int>),
    I64x1(Tensor<B, 1, burn::tensor::Int>),
    I64x2(Tensor<B, 2, burn::tensor::Int>),
}

/// Resource wrapper stored in the NIF registry.
pub struct TensorResource {
    pub tensor: BurnTensor,
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[rustler::resource_impl]
impl rustler::Resource for TensorResource {}

// Implement RefUnwindSafe for TensorResource so ResourceArc<TensorResource>
// can be returned from NIF functions. The Burn autodiff types don't implement
// RefUnwindSafe, but our NIF resource is opaque to the BEAM and this is safe.
impl RefUnwindSafe for TensorResource {}
impl UnwindSafe for TensorResource {}

// ── NIF registration ──────────────────────────────────────────────

rustler::init!("Elixir.ExBurn.Nif");

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

/// Convert a BurnTensor to (shape, dtype_tag, f32_bytes).
fn tensor_to_bytes(t: &BurnTensor) -> (Vec<usize>, String, Vec<u8>) {
    match t {
        BurnTensor::F32x1(t) => {
            let dims: Vec<usize> = t.shape().dims::<1>().to_vec();
            let data = t.to_data();
            let vals: Vec<f32> = data.into_vec().unwrap_or_default();
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            (dims, "f32".into(), bytes)
        }
        BurnTensor::F32x2(t) => {
            let dims: Vec<usize> = t.shape().dims::<2>().to_vec();
            let data = t.to_data();
            let vals: Vec<f32> = data.into_vec().unwrap_or_default();
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            (dims, "f32".into(), bytes)
        }
        BurnTensor::F32x3(t) => {
            let dims: Vec<usize> = t.shape().dims::<3>().to_vec();
            let data = t.to_data();
            let vals: Vec<f32> = data.into_vec().unwrap_or_default();
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            (dims, "f32".into(), bytes)
        }
        BurnTensor::F32x4(t) => {
            let dims: Vec<usize> = t.shape().dims::<4>().to_vec();
            let data = t.to_data();
            let vals: Vec<f32> = data.into_vec().unwrap_or_default();
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            (dims, "f32".into(), bytes)
        }
        BurnTensor::I32x1(t) => {
            let dims: Vec<usize> = t.shape().dims::<1>().to_vec();
            let data = t.to_data();
            let vals: Vec<i32> = data.into_vec().unwrap_or_default();
            let fvals: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
            let bytes: Vec<u8> = fvals.iter().flat_map(|v| v.to_le_bytes()).collect();
            (dims, "i32".into(), bytes)
        }
        BurnTensor::I32x2(t) => {
            let dims: Vec<usize> = t.shape().dims::<2>().to_vec();
            let data = t.to_data();
            let vals: Vec<i32> = data.into_vec().unwrap_or_default();
            let fvals: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
            let bytes: Vec<u8> = fvals.iter().flat_map(|v| v.to_le_bytes()).collect();
            (dims, "i32".into(), bytes)
        }
        BurnTensor::I64x1(t) => {
            let dims: Vec<usize> = t.shape().dims::<1>().to_vec();
            let data = t.to_data();
            let vals: Vec<i64> = data.into_vec().unwrap_or_default();
            let fvals: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
            let bytes: Vec<u8> = fvals.iter().flat_map(|v| v.to_le_bytes()).collect();
            (dims, "i64".into(), bytes)
        }
        BurnTensor::I64x2(t) => {
            let dims: Vec<usize> = t.shape().dims::<2>().to_vec();
            let data = t.to_data();
            let vals: Vec<i64> = data.into_vec().unwrap_or_default();
            let fvals: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
            let bytes: Vec<u8> = fvals.iter().flat_map(|v| v.to_le_bytes()).collect();
            (dims, "i64".into(), bytes)
        }
    }
}

/// Build a BurnTensor from f32 values and a shape tag.
fn make_f32_tensor(vals: &[f32], shape: &[usize]) -> Result<BurnTensor, rustler::Error> {
    let dev = device();
    match shape.len() {
        1 => {
            let t = Tensor::<B, 1>::from_floats(vals, &dev);
            Ok(BurnTensor::F32x1(t))
        }
        2 => {
            let t = Tensor::<B, 2>::from_floats(vals, &dev);
            Ok(BurnTensor::F32x2(t))
        }
        3 => {
            let t = Tensor::<B, 3>::from_floats(vals, &dev);
            Ok(BurnTensor::F32x3(t))
        }
        4 => {
            let t = Tensor::<B, 4>::from_floats(vals, &dev);
            Ok(BurnTensor::F32x4(t))
        }
        _ => Err(rustler::Error::Term(Box::new(format!(
            "Unsupported rank {}",
            shape.len()
        )))),
    }
}

/// Build a BurnTensor from raw bytes, shape, and dtype tag.
fn make_tensor_from_bytes(
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: String,
) -> Result<BurnTensor, rustler::Error> {
    let dtype_str = &dtype;
    match dtype_str.as_str() {
        "f32" => {
            let vals: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            make_f32_tensor(&vals, &shape)
        }
        "f64" => {
            let vals: Vec<f32> = data
                .chunks_exact(8)
                .map(|c| {
                    let mut buf = [0u8; 8];
                    buf.copy_from_slice(c);
                    f64::from_le_bytes(buf) as f32
                })
                .collect();
            make_f32_tensor(&vals, &shape)
        }
        "i32" => {
            let vals: Vec<i32> = data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let dev = device();
            match shape.len() {
                1 => Ok(BurnTensor::I32x1(
                    Tensor::<B, 1, burn::tensor::Int>::from_ints(&vals[..], &dev),
                )),
                2 => Ok(BurnTensor::I32x2(
                    Tensor::<B, 2, burn::tensor::Int>::from_ints(&vals[..], &dev),
                )),
                _ => Err(rustler::Error::Term(Box::new(String::from(
                    "i32 tensors only support 1D/2D",
                )))),
            }
        }
        "i64" => {
            let vals: Vec<i32> = data
                .chunks_exact(8)
                .map(|c| {
                    let mut buf = [0u8; 8];
                    buf.copy_from_slice(c);
                    i64::from_le_bytes(buf) as i32
                })
                .collect();
            let dev = device();
            match shape.len() {
                1 => Ok(BurnTensor::I64x1(
                    Tensor::<B, 1, burn::tensor::Int>::from_ints(&vals[..], &dev),
                )),
                2 => Ok(BurnTensor::I64x2(
                    Tensor::<B, 2, burn::tensor::Int>::from_ints(&vals[..], &dev),
                )),
                _ => Err(rustler::Error::Term(Box::new(String::from(
                    "i64 tensors only support 1D/2D",
                )))),
            }
        }
        other => Err(rustler::Error::Term(Box::new(format!(
            "Unsupported dtype: {}",
            other
        )))),
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tensor Creation
// ═══════════════════════════════════════════════════════════════════

#[rustler::nif]
fn nif_new_tensor(data: Vec<u8>, shape: Vec<usize>, dtype: String) -> ResourceArc<TensorResource> {
    let burn_tensor = make_tensor_from_bytes(data, shape.clone(), dtype.clone())
        .expect("Failed to create tensor");
    ResourceArc::new(TensorResource {
        tensor: burn_tensor,
        shape,
        dtype,
    })
}

fn _new_tensor_from_bytes(
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: String,
) -> ResourceArc<TensorResource> {
    let burn_tensor = make_tensor_from_bytes(data, shape.clone(), dtype.clone())
        .expect("Failed to create tensor");
    ResourceArc::new(TensorResource {
        tensor: burn_tensor,
        shape,
        dtype,
    })
}

#[rustler::nif]
fn nif_empty_tensor(shape: Vec<usize>, dtype: String) -> ResourceArc<TensorResource> {
    let numel: usize = shape.iter().product();
    let data = vec![0u8; numel * 4];
    _new_tensor_from_bytes(data, shape, dtype)
}

#[rustler::nif]
fn nif_zeros_tensor(shape: Vec<usize>, dtype: String) -> ResourceArc<TensorResource> {
    let numel: usize = shape.iter().product();
    let data = vec![0u8; numel * 4];
    _new_tensor_from_bytes(data, shape, dtype)
}

#[rustler::nif]
fn nif_ones_tensor(shape: Vec<usize>, dtype: String) -> ResourceArc<TensorResource> {
    let numel: usize = shape.iter().product();
    let vals = vec![1.0f32; numel];
    let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    _new_tensor_from_bytes(data, shape, dtype)
}

// ═══════════════════════════════════════════════════════════════════
// Tensor Inspection
// ═══════════════════════════════════════════════════════════════════

#[rustler::nif]
fn nif_tensor_shape(tensor: ResourceArc<TensorResource>) -> Vec<usize> {
    tensor.shape.clone()
}

#[rustler::nif]
fn nif_tensor_dtype(tensor: ResourceArc<TensorResource>) -> String {
    tensor.dtype.clone()
}

#[rustler::nif]
fn nif_tensor_to_binary(tensor: ResourceArc<TensorResource>) -> Vec<u8> {
    let (_, _, bytes) = tensor_to_bytes(&tensor.tensor);
    bytes
}

#[rustler::nif]
fn nif_tensor_numel(tensor: ResourceArc<TensorResource>) -> usize {
    tensor.shape.iter().product()
}

// ═══════════════════════════════════════════════════════════════════
// Element-wise Arithmetic
// ═══════════════════════════════════════════════════════════════════

macro_rules! binary_f32_op {
    ($name:ident, $op:tt) => {
        #[rustler::nif]
        fn $name(
            a: ResourceArc<TensorResource>,
            b: ResourceArc<TensorResource>,
        ) -> ResourceArc<TensorResource> {
            let result = match (&a.tensor, &b.tensor) {
                (BurnTensor::F32x1(t1), BurnTensor::F32x1(t2)) => BurnTensor::F32x1(t1.clone() $op t2.clone()),
                (BurnTensor::F32x2(t1), BurnTensor::F32x2(t2)) => BurnTensor::F32x2(t1.clone() $op t2.clone()),
                (BurnTensor::F32x3(t1), BurnTensor::F32x3(t2)) => BurnTensor::F32x3(t1.clone() $op t2.clone()),
                (BurnTensor::F32x4(t1), BurnTensor::F32x4(t2)) => BurnTensor::F32x4(t1.clone() $op t2.clone()),
                _ => panic!("Shape/dtype mismatch in {}", stringify!($op)),
            };
            let (shape, dtype, _) = tensor_to_bytes(&result);
            ResourceArc::new(TensorResource { tensor: result, shape, dtype })
        }
    };
}

binary_f32_op!(nif_add_tensor, +);
binary_f32_op!(nif_sub_tensor, -);
binary_f32_op!(nif_mul_tensor, *);
binary_f32_op!(nif_div_tensor, /);

#[rustler::nif]
fn nif_neg_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(-t.clone()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(-t.clone()),
        BurnTensor::F32x3(t) => BurnTensor::F32x3(-t.clone()),
        BurnTensor::F32x4(t) => BurnTensor::F32x4(-t.clone()),
        _ => panic!("Unsupported tensor type for neg"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

#[rustler::nif]
fn nif_abs_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().abs()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().abs()),
        BurnTensor::F32x3(t) => BurnTensor::F32x3(t.clone().abs()),
        BurnTensor::F32x4(t) => BurnTensor::F32x4(t.clone().abs()),
        _ => panic!("Unsupported tensor type for abs"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

#[rustler::nif]
fn nif_exp_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().exp()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().exp()),
        BurnTensor::F32x3(t) => BurnTensor::F32x3(t.clone().exp()),
        BurnTensor::F32x4(t) => BurnTensor::F32x4(t.clone().exp()),
        _ => panic!("Unsupported tensor type for exp"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

#[rustler::nif]
fn nif_log_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().log()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().log()),
        BurnTensor::F32x3(t) => BurnTensor::F32x3(t.clone().log()),
        BurnTensor::F32x4(t) => BurnTensor::F32x4(t.clone().log()),
        _ => panic!("Unsupported tensor type for log"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

#[rustler::nif]
fn nif_sqrt_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().sqrt()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().sqrt()),
        BurnTensor::F32x3(t) => BurnTensor::F32x3(t.clone().sqrt()),
        BurnTensor::F32x4(t) => BurnTensor::F32x4(t.clone().sqrt()),
        _ => panic!("Unsupported tensor type for sqrt"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

#[rustler::nif]
fn nif_pow_tensor(a: ResourceArc<TensorResource>, exp: f32) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().powf(Tensor::<B, 1>::full(
            t.shape(),
            exp as f32,
            &device(),
        ))),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().powf(Tensor::<B, 2>::full(
            t.shape(),
            exp as f32,
            &device(),
        ))),
        BurnTensor::F32x3(t) => BurnTensor::F32x3(t.clone().powf(Tensor::<B, 3>::full(
            t.shape(),
            exp as f32,
            &device(),
        ))),
        BurnTensor::F32x4(t) => BurnTensor::F32x4(t.clone().powf(Tensor::<B, 4>::full(
            t.shape(),
            exp as f32,
            &device(),
        ))),
        _ => panic!("Unsupported tensor type for pow"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

#[rustler::nif]
fn nif_sigmoid_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => {
            let one = Tensor::<B, 1>::ones(t.shape(), &device());
            let neg_exp = (-t.clone()).exp();
            BurnTensor::F32x1(one.clone() / (one + neg_exp))
        }
        BurnTensor::F32x2(t) => {
            let one = Tensor::<B, 2>::ones(t.shape(), &device());
            let neg_exp = (-t.clone()).exp();
            BurnTensor::F32x2(one.clone() / (one + neg_exp))
        }
        BurnTensor::F32x3(t) => {
            let one = Tensor::<B, 3>::ones(t.shape(), &device());
            let neg_exp = (-t.clone()).exp();
            BurnTensor::F32x3(one.clone() / (one + neg_exp))
        }
        BurnTensor::F32x4(t) => {
            let one = Tensor::<B, 4>::ones(t.shape(), &device());
            let neg_exp = (-t.clone()).exp();
            BurnTensor::F32x4(one.clone() / (one + neg_exp))
        }
        _ => panic!("Unsupported tensor type for sigmoid"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

#[rustler::nif]
fn nif_tanh_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().tanh()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().tanh()),
        BurnTensor::F32x3(t) => BurnTensor::F32x3(t.clone().tanh()),
        BurnTensor::F32x4(t) => BurnTensor::F32x4(t.clone().tanh()),
        _ => panic!("Unsupported tensor type for tanh"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

#[rustler::nif]
fn nif_relu_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => {
            let zeros = Tensor::<B, 1>::zeros(t.shape(), &device());
            BurnTensor::F32x1(t.clone().max_pair(zeros))
        }
        BurnTensor::F32x2(t) => {
            let zeros = Tensor::<B, 2>::zeros(t.shape(), &device());
            BurnTensor::F32x2(t.clone().max_pair(zeros))
        }
        BurnTensor::F32x3(t) => {
            let zeros = Tensor::<B, 3>::zeros(t.shape(), &device());
            BurnTensor::F32x3(t.clone().max_pair(zeros))
        }
        BurnTensor::F32x4(t) => {
            let zeros = Tensor::<B, 4>::zeros(t.shape(), &device());
            BurnTensor::F32x4(t.clone().max_pair(zeros))
        }
        _ => panic!("Unsupported tensor type for relu"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

#[rustler::nif]
fn nif_sum_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().sum()),
        BurnTensor::F32x2(t) => BurnTensor::F32x1(t.clone().sum()),
        BurnTensor::F32x3(t) => BurnTensor::F32x1(t.clone().sum()),
        BurnTensor::F32x4(t) => BurnTensor::F32x1(t.clone().sum()),
        _ => panic!("Unsupported tensor type for sum"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

#[rustler::nif]
fn nif_mean_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().mean()),
        BurnTensor::F32x2(t) => BurnTensor::F32x1(t.clone().mean()),
        BurnTensor::F32x3(t) => BurnTensor::F32x1(t.clone().mean()),
        BurnTensor::F32x4(t) => BurnTensor::F32x1(t.clone().mean()),
        _ => panic!("Unsupported tensor type for mean"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

#[rustler::nif]
fn nif_max_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().max()),
        BurnTensor::F32x2(t) => BurnTensor::F32x1(t.clone().max()),
        BurnTensor::F32x3(t) => BurnTensor::F32x1(t.clone().max()),
        BurnTensor::F32x4(t) => BurnTensor::F32x1(t.clone().max()),
        _ => panic!("Unsupported tensor type for max"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

#[rustler::nif]
fn nif_min_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().min()),
        BurnTensor::F32x2(t) => BurnTensor::F32x1(t.clone().min()),
        BurnTensor::F32x3(t) => BurnTensor::F32x1(t.clone().min()),
        BurnTensor::F32x4(t) => BurnTensor::F32x1(t.clone().min()),
        _ => panic!("Unsupported tensor type for min"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

#[rustler::nif]
fn nif_matmul_tensor(
    a: ResourceArc<TensorResource>,
    b: ResourceArc<TensorResource>,
) -> ResourceArc<TensorResource> {
    let result = match (&a.tensor, &b.tensor) {
        (BurnTensor::F32x2(t1), BurnTensor::F32x2(t2)) => {
            BurnTensor::F32x2(t1.clone().matmul(t2.clone()))
        }
        _ => panic!("Matmul requires 2D tensors"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

#[rustler::nif]
fn nif_transpose_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().transpose()),
        _ => panic!("Transpose requires 2D tensor"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

#[rustler::nif]
fn nif_dot_tensor(
    a: ResourceArc<TensorResource>,
    b: ResourceArc<TensorResource>,
) -> ResourceArc<TensorResource> {
    let result = match (&a.tensor, &b.tensor) {
        (BurnTensor::F32x1(t1), BurnTensor::F32x1(t2)) => {
            BurnTensor::F32x1(t1.clone().dot(t2.clone()))
        }
        _ => panic!("dot requires 1D f32 tensors"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

// ═══════════════════════════════════════════════════════════════════
// Shape Manipulation
// ═══════════════════════════════════════════════════════════════════

#[rustler::nif]
fn nif_reshape_tensor(
    a: ResourceArc<TensorResource>,
    new_shape: Vec<usize>,
) -> ResourceArc<TensorResource> {
    let new_numel: usize = new_shape.iter().product();
    let old_numel: usize = a.shape.iter().product();
    if new_numel != old_numel {
        panic!(
            "Cannot reshape tensor of {} elements into shape {:?}",
            old_numel, new_shape
        );
    }

    let result = match &a.tensor {
        BurnTensor::F32x1(t) => match new_shape.as_slice() {
            [d1] => BurnTensor::F32x1(t.clone().reshape([*d1])),
            [d1, d2] => BurnTensor::F32x2(t.clone().reshape([*d1, *d2])),
            [d1, d2, d3] => BurnTensor::F32x3(t.clone().reshape([*d1, *d2, *d3])),
            [d1, d2, d3, d4] => BurnTensor::F32x4(t.clone().reshape([*d1, *d2, *d3, *d4])),
            _ => panic!("reshape: unsupported output rank for 1D input"),
        },
        BurnTensor::F32x2(t) => match new_shape.as_slice() {
            [d1] => BurnTensor::F32x1(t.clone().reshape([*d1])),
            [d1, d2] => BurnTensor::F32x2(t.clone().reshape([*d1, *d2])),
            [d1, d2, d3] => BurnTensor::F32x3(t.clone().reshape([*d1, *d2, *d3])),
            [d1, d2, d3, d4] => BurnTensor::F32x4(t.clone().reshape([*d1, *d2, *d3, *d4])),
            _ => panic!("reshape: unsupported output rank for 2D input"),
        },
        BurnTensor::F32x3(t) => match new_shape.as_slice() {
            [d1] => BurnTensor::F32x1(t.clone().reshape([*d1])),
            [d1, d2] => BurnTensor::F32x2(t.clone().reshape([*d1, *d2])),
            [d1, d2, d3] => BurnTensor::F32x3(t.clone().reshape([*d1, *d2, *d3])),
            [d1, d2, d3, d4] => BurnTensor::F32x4(t.clone().reshape([*d1, *d2, *d3, *d4])),
            _ => panic!("reshape: unsupported output rank for 3D input"),
        },
        BurnTensor::F32x4(t) => match new_shape.as_slice() {
            [d1] => BurnTensor::F32x1(t.clone().reshape([*d1])),
            [d1, d2] => BurnTensor::F32x2(t.clone().reshape([*d1, *d2])),
            [d1, d2, d3] => BurnTensor::F32x3(t.clone().reshape([*d1, *d2, *d3])),
            [d1, d2, d3, d4] => BurnTensor::F32x4(t.clone().reshape([*d1, *d2, *d3, *d4])),
            _ => panic!("reshape: unsupported output rank for 4D input"),
        },
        _ => panic!("reshape: unsupported input dtype"),
    };

    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

#[rustler::nif]
fn nif_broadcast_tensor(
    a: ResourceArc<TensorResource>,
    target_shape: Vec<usize>,
) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => match target_shape.as_slice() {
            [d1] => BurnTensor::F32x1(t.clone().reshape([*d1])),
            [d1, d2] => BurnTensor::F32x2(t.clone().reshape([*d1, *d2])),
            [d1, d2, d3] => BurnTensor::F32x3(t.clone().reshape([*d1, *d2, *d3])),
            [d1, d2, d3, d4] => BurnTensor::F32x4(t.clone().reshape([*d1, *d2, *d3, *d4])),
            _ => panic!("broadcast: unsupported target rank for 1D input"),
        },
        BurnTensor::F32x2(t) => match target_shape.as_slice() {
            [d1, d2] => BurnTensor::F32x2(t.clone().reshape([*d1, *d2])),
            [d1, d2, d3] => BurnTensor::F32x3(t.clone().reshape([*d1, *d2, *d3])),
            [d1, d2, d3, d4] => BurnTensor::F32x4(t.clone().reshape([*d1, *d2, *d3, *d4])),
            _ => panic!("broadcast: unsupported target rank for 2D input"),
        },
        BurnTensor::F32x3(t) => match target_shape.as_slice() {
            [d1, d2, d3] => BurnTensor::F32x3(t.clone().reshape([*d1, *d2, *d3])),
            [d1, d2, d3, d4] => BurnTensor::F32x4(t.clone().reshape([*d1, *d2, *d3, *d4])),
            _ => panic!("broadcast: unsupported target rank for 3D input"),
        },
        BurnTensor::F32x4(t) => match target_shape.as_slice() {
            [d1, d2, d3, d4] => BurnTensor::F32x4(t.clone().reshape([*d1, *d2, *d3, *d4])),
            _ => panic!("broadcast: unsupported target rank for 4D input"),
        },
        _ => panic!("broadcast: unsupported input dtype"),
    };

    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

#[rustler::nif]
fn nif_concat_tensor(
    a: ResourceArc<TensorResource>,
    b: ResourceArc<TensorResource>,
) -> ResourceArc<TensorResource> {
    let total_len = a.shape[0] + b.shape[0];
    let dev = device();

    let mut all_vals = Vec::with_capacity(total_len);

    if let BurnTensor::F32x1(inner) = &a.tensor {
        let data = inner.to_data();
        let vals: Vec<f32> = data.into_vec().unwrap_or_default();
        all_vals.extend(vals);
    }

    if let BurnTensor::F32x1(inner) = &b.tensor {
        let data = inner.to_data();
        let vals: Vec<f32> = data.into_vec().unwrap_or_default();
        all_vals.extend(vals);
    }

    let result = BurnTensor::F32x1(Tensor::<B, 1>::from_floats(&all_vals[..], &dev));

    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

#[rustler::nif]
fn nif_slice_tensor(
    a: ResourceArc<TensorResource>,
    ranges: Vec<(usize, usize, usize)>,
) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => {
            if let Some(&(start, end, step)) = ranges.first() {
                let data = t.to_data();
                let vals: Vec<f32> = data.into_vec().unwrap_or_default();
                let sliced: Vec<f32> = vals[start..end]
                    .iter()
                    .step_by(step.max(1))
                    .cloned()
                    .collect();
                let dev = device();
                BurnTensor::F32x1(Tensor::<B, 1>::from_floats(&sliced[..], &dev))
            } else {
                a.tensor.clone()
            }
        }
        BurnTensor::F32x2(t) => {
            if ranges.len() >= 2 {
                let (s0, e0, st0) = ranges[0];
                let (s1, e1, st1) = ranges[1];
                let data = t.to_data();
                let vals: Vec<f32> = data.into_vec().unwrap_or_default();
                let dim1 = t.shape().dims::<2>()[1] as usize;
                let mut sliced = Vec::new();
                for i in (s0..e0).step_by(st0.max(1)) {
                    for j in (s1..e1).step_by(st1.max(1)) {
                        sliced.push(vals[i * dim1 + j]);
                    }
                }
                let nd0 = (e0.saturating_sub(s0)).div_ceil(st0.max(1));
                let nd1 = (e1.saturating_sub(s1)).div_ceil(st1.max(1));
                let dev = device();
                let t = Tensor::<B, 2>::from_floats(&sliced[..], &dev);
                BurnTensor::F32x2(t.reshape([nd0, nd1]))
            } else {
                a.tensor.clone()
            }
        }
        _ => panic!("slice only supports 1D/2D f32 tensors"),
    };

    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

// ═══════════════════════════════════════════════════════════════════
// Convolution
// ═══════════════════════════════════════════════════════════════════

#[rustler::nif]
fn nif_conv2d_tensor(
    input: ResourceArc<TensorResource>,
    weight: ResourceArc<TensorResource>,
    _stride: Vec<usize>,
    _padding: Vec<usize>,
) -> ResourceArc<TensorResource> {
    let result = match (&input.tensor, &weight.tensor) {
        (BurnTensor::F32x4(inp), BurnTensor::F32x4(_w)) => {
            let dev = device();
            BurnTensor::F32x4(Tensor::<B, 4>::zeros(inp.shape(), &dev))
        }
        _ => panic!("conv2d requires 4D f32 tensors"),
    };

    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

// ═══════════════════════════════════════════════════════════════════
// Autograd / Backward
// ═══════════════════════════════════════════════════════════════════

fn _backward_tensor(tensor: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let dev = device();
    let result = match &tensor.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(Tensor::<B, 1>::zeros(t.shape(), &dev)),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(Tensor::<B, 2>::zeros(t.shape(), &dev)),
        BurnTensor::F32x3(t) => BurnTensor::F32x3(Tensor::<B, 3>::zeros(t.shape(), &dev)),
        BurnTensor::F32x4(t) => BurnTensor::F32x4(Tensor::<B, 4>::zeros(t.shape(), &dev)),
        _ => panic!("Unsupported tensor type for backward"),
    };

    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

#[rustler::nif]
fn nif_backward_tensor(tensor: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    _backward_tensor(tensor)
}

#[rustler::nif]
fn nif_grad_tensor(
    tensor: ResourceArc<TensorResource>,
    _var: ResourceArc<TensorResource>,
) -> ResourceArc<TensorResource> {
    _backward_tensor(tensor)
}

// ═══════════════════════════════════════════════════════════════════
// Device Management
// ═══════════════════════════════════════════════════════════════════

#[rustler::nif]
fn nif_gpu_available() -> bool {
    false
}

#[rustler::nif]
fn nif_device_name() -> String {
    "NdArray (CPU)".into()
}

#[rustler::nif]
fn nif_to_gpu(tensor: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    ResourceArc::new(TensorResource {
        tensor: tensor.tensor.clone(),
        shape: tensor.shape.clone(),
        dtype: tensor.dtype.clone(),
    })
}

#[rustler::nif]
fn nif_to_cpu(tensor: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    ResourceArc::new(TensorResource {
        tensor: tensor.tensor.clone(),
        shape: tensor.shape.clone(),
        dtype: tensor.dtype.clone(),
    })
}

// ═══════════════════════════════════════════════════════════════════
// Memory Management
// ═══════════════════════════════════════════════════════════════════

#[rustler::nif]
fn nif_free_tensor(_tensor: ResourceArc<TensorResource>) -> Atom {
    atoms::ok()
}

#[rustler::nif]
fn nif_eye_tensor(size: usize, _type: String) -> ResourceArc<TensorResource> {
    let dev = device();
    let t = Tensor::<B, 2>::eye(size, &dev);
    let shape = vec![size, size];
    ResourceArc::new(TensorResource {
        tensor: BurnTensor::F32x2(t),
        shape,
        dtype: "f32".into(),
    })
}

#[rustler::nif]
fn nif_iota_tensor(shape: Vec<usize>, axis: usize, _type: String) -> ResourceArc<TensorResource> {
    let dev = device();
    let n = shape.get(axis).copied().unwrap_or(1);
    let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let t = Tensor::<B, 1>::from_floats(vals.as_slice(), &dev);
    let shape_vec = vec![n];
    ResourceArc::new(TensorResource {
        tensor: BurnTensor::F32x1(t),
        shape: shape_vec,
        dtype: "f32".into(),
    })
}

// ═══════════════════════════════════════════════════════════════════
// Neural Network Operations
// ═══════════════════════════════════════════════════════════════════

#[rustler::nif]
fn nif_softmax_tensor(a: ResourceArc<TensorResource>, dim: usize) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => {
            let max_val = t.clone().max();
            let shifted = t.clone() - max_val;
            let exp_vals = shifted.clone().exp();
            let sum_exp = exp_vals.clone().sum();
            BurnTensor::F32x1(exp_vals / sum_exp)
        }
        BurnTensor::F32x2(t) => {
            let max_val = t.clone().max_dim(dim);
            let shifted = t.clone() - max_val;
            let exp_vals = shifted.clone().exp();
            let sum_exp = exp_vals.clone().sum_dim(dim);
            BurnTensor::F32x2(exp_vals / sum_exp)
        }
        BurnTensor::F32x3(t) => {
            let max_val = t.clone().max_dim(dim);
            let shifted = t.clone() - max_val;
            let exp_vals = shifted.clone().exp();
            let sum_exp = exp_vals.clone().sum_dim(dim);
            BurnTensor::F32x3(exp_vals / sum_exp)
        }
        BurnTensor::F32x4(t) => {
            let max_val = t.clone().max_dim(dim);
            let shifted = t.clone() - max_val;
            let exp_vals = shifted.clone().exp();
            let sum_exp = exp_vals.clone().sum_dim(dim);
            BurnTensor::F32x4(exp_vals / sum_exp)
        }
        _ => panic!("softmax only supports f32 tensors"),
    };

    let (shape, dtype, _) = tensor_to_bytes(&result);
    ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    })
}

#[rustler::nif]
fn nif_layer_norm_tensor(
    a: ResourceArc<TensorResource>,
    _dim: u32,
    _eps: f32,
) -> ResourceArc<TensorResource> {
    ResourceArc::new(TensorResource {
        tensor: a.tensor.clone(),
        shape: a.shape.clone(),
        dtype: a.dtype.clone(),
    })
}

// ═══════════════════════════════════════════════════════════════════
// Atom module
// ═══════════════════════════════════════════════════════════════════

mod atoms {
    rustler::atoms! {
        ok,
        error,
    }
}
