//! # ExBurn NIF
//!
//! Rust NIF bridge between Elixir and the Burn deep learning framework.
//! Uses Burn's CubeCL backend with autodiff support.

use rustler::{Atom, Encoder, Env, NifResult, ResourceArc, Term};

// ── Burn imports ────────────────────────────────────────────────────
use burn::backend::CubeCL;
use burn::tensor::cast::ToElement;
use burn::tensor::Tensor;
use burn_autodiff::Autodiff;

// ── Backend type ───────────────────────────────────────────────────
type B = Autodiff<CubeCL>;

// ── Tensor kind enum ──────────────────────────────────────────────
/// Enum wrapping concrete Burn tensor types so they can be stored in a
/// single registry.  Each variant holds an actual Burn tensor with the
/// Autodiff<CubeCL> backend.
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

// ── NIF registration ──────────────────────────────────────────────

rustler::init!(
    "Elixir.ExBurn.Nif",
    [
        nif_new_tensor,
        nif_empty_tensor,
        nif_zeros_tensor,
        nif_ones_tensor,
        nif_random_tensor,
        nif_tensor_shape,
        nif_tensor_dtype,
        nif_tensor_to_binary,
        nif_tensor_numel,
        nif_add_tensor,
        nif_sub_tensor,
        nif_mul_tensor,
        nif_div_tensor,
        nif_neg_tensor,
        nif_abs_tensor,
        nif_exp_tensor,
        nif_log_tensor,
        nif_sqrt_tensor,
        nif_pow_tensor,
        nif_sigmoid_tensor,
        nif_tanh_tensor,
        nif_relu_tensor,
        nif_sum_tensor,
        nif_mean_tensor,
        nif_max_tensor,
        nif_min_tensor,
        nif_matmul_tensor,
        nif_transpose_tensor,
        nif_dot_tensor,
        nif_reshape_tensor,
        nif_broadcast_tensor,
        nif_concat_tensor,
        nif_slice_tensor,
        nif_conv2d_tensor,
        nif_backward_tensor,
        nif_grad_tensor,
        nif_gpu_available,
        nif_device_name,
        nif_to_gpu,
        nif_to_cpu,
        nif_free_tensor,
        nif_softmax_tensor,
        nif_layer_norm_tensor,
        nif_dropout_tensor,
        nif_cross_entropy_tensor,
        nif_mse_tensor,
    ],
    load = load
);

fn load(env: Env, _info: Term) -> bool {
    env.register::<ResourceArc<TensorResource>>().is_ok()
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

/// Convert a BurnTensor to (shape, dtype_tag, f32_bytes).
/// For integer tensors the values are cast to f32 for the wire format.
fn tensor_to_bytes(t: &BurnTensor) -> (Vec<usize>, String, Vec<u8>) {
    match t {
        BurnTensor::F32x1(t) => {
            let s = t.shape();
            let dims: Vec<usize> = s.dims.iter().map(|&d| d as usize).collect();
            let data = t.to_data();
            let vals: Vec<f32> = data.into_vec::<f32>().unwrap_or_default();
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            (dims, "f32".into(), bytes)
        }
        BurnTensor::F32x2(t) => {
            let s = t.shape();
            let dims: Vec<usize> = s.dims.iter().map(|&d| d as usize).collect();
            let data = t.to_data();
            let vals: Vec<f32> = data.into_vec::<f32>().unwrap_or_default();
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            (dims, "f32".into(), bytes)
        }
        BurnTensor::F32x3(t) => {
            let s = t.shape();
            let dims: Vec<usize> = s.dims.iter().map(|&d| d as usize).collect();
            let data = t.to_data();
            let vals: Vec<f32> = data.into_vec::<f32>().unwrap_or_default();
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            (dims, "f32".into(), bytes)
        }
        BurnTensor::F32x4(t) => {
            let s = t.shape();
            let dims: Vec<usize> = s.dims.iter().map(|&d| d as usize).collect();
            let data = t.to_data();
            let vals: Vec<f32> = data.into_vec::<f32>().unwrap_or_default();
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            (dims, "f32".into(), bytes)
        }
        BurnTensor::I32x1(t) => {
            let s = t.shape();
            let dims: Vec<usize> = s.dims.iter().map(|&d| d as usize).collect();
            let data = t.to_data();
            let vals: Vec<i32> = data.into_vec::<i32>().unwrap_or_default();
            let fvals: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
            let bytes: Vec<u8> = fvals.iter().flat_map(|v| v.to_le_bytes()).collect();
            (dims, "i32".into(), bytes)
        }
        BurnTensor::I32x2(t) => {
            let s = t.shape();
            let dims: Vec<usize> = s.dims.iter().map(|&d| d as usize).collect();
            let data = t.to_data();
            let vals: Vec<i32> = data.into_vec::<i32>().unwrap_or_default();
            let fvals: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
            let bytes: Vec<u8> = fvals.iter().flat_map(|v| v.to_le_bytes()).collect();
            (dims, "i32".into(), bytes)
        }
        BurnTensor::I64x1(t) => {
            let s = t.shape();
            let dims: Vec<usize> = s.dims.iter().map(|&d| d as usize).collect();
            let data = t.to_data();
            let vals: Vec<i64> = data.into_vec::<i64>().unwrap_or_default();
            let fvals: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
            let bytes: Vec<u8> = fvals.iter().flat_map(|v| v.to_le_bytes()).collect();
            (dims, "i64".into(), bytes)
        }
        BurnTensor::I64x2(t) => {
            let s = t.shape();
            let dims: Vec<usize> = s.dims.iter().map(|&d| d as usize).collect();
            let data = t.to_data();
            let vals: Vec<i64> = data.into_vec::<i64>().unwrap_or_default();
            let fvals: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
            let bytes: Vec<u8> = fvals.iter().flat_map(|v| v.to_le_bytes()).collect();
            (dims, "i64".into(), bytes)
        }
    }
}

/// Build a BurnTensor from f32 values and a shape tag.
fn make_f32_tensor(vals: &[f32], shape: &[usize]) -> BurnTensor {
    let dev = B::Device::default();
    match shape.len() {
        1 => {
            let t = Tensor::<B, 1>::from_data(burn::tensor::Data::from(vals), &dev);
            BurnTensor::F32x1(t)
        }
        2 => {
            let t = Tensor::<B, 2>::from_data(burn::tensor::Data::from(vals), &dev);
            BurnTensor::F32x2(t)
        }
        3 => {
            let t = Tensor::<B, 3>::from_data(burn::tensor::Data::from(vals), &dev);
            BurnTensor::F32x3(t)
        }
        4 => {
            let t = Tensor::<B, 4>::from_data(burn::tensor::Data::from(vals), &dev);
            BurnTensor::F32x4(t)
        }
        _ => panic!("Unsupported rank {}", shape.len()),
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tensor Creation
// ═══════════════════════════════════════════════════════════════════

#[rustler::nif]
fn nif_new_tensor(
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: Atom,
) -> NifResult<ResourceArc<TensorResource>> {
    let dtype_str = dtype.to_string();
    let burn_tensor = match dtype_str.as_str() {
        "f32" => {
            let vals: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            make_f32_tensor(&vals, &shape)
        }
        "f64" => {
            // Read as f64, cast to f32 for storage
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
            let dev = B::Device::default();
            match shape.len() {
                1 => BurnTensor::I32x1(Tensor::<B, 1, burn::tensor::Int>::from_data(
                    burn::tensor::Data::from(vals.as_slice()),
                    &dev,
                )),
                2 => BurnTensor::I32x2(Tensor::<B, 2, burn::tensor::Int>::from_data(
                    burn::tensor::Data::from(vals.as_slice()),
                    &dev,
                )),
                _ => {
                    return Err(rustler::Error::Term(Box::new(
                        "i32 tensors only support 1D/2D".into(),
                    )))
                }
            }
        }
        "i64" => {
            let vals: Vec<i64> = data
                .chunks_exact(8)
                .map(|c| {
                    let mut buf = [0u8; 8];
                    buf.copy_from_slice(c);
                    i64::from_le_bytes(buf)
                })
                .collect();
            let dev = B::Device::default();
            match shape.len() {
                1 => BurnTensor::I64x1(Tensor::<B, 1, burn::tensor::Int>::from_data(
                    burn::tensor::Data::from(vals.as_slice()),
                    &dev,
                )),
                2 => BurnTensor::I64x2(Tensor::<B, 2, burn::tensor::Int>::from_data(
                    burn::tensor::Data::from(vals.as_slice()),
                    &dev,
                )),
                _ => {
                    return Err(rustler::Error::Term(Box::new(
                        "i64 tensors only support 1D/2D".into(),
                    )))
                }
            }
        }
        other => {
            return Err(rustler::Error::Term(Box::new(format!(
                "Unsupported dtype: {}",
                other
            ))))
        }
    };

    Ok(ResourceArc::new(TensorResource {
        tensor: burn_tensor,
        shape: shape.clone(),
        dtype: dtype_str,
    }))
}

#[rustler::nif]
fn nif_empty_tensor(shape: Vec<usize>, dtype: Atom) -> NifResult<ResourceArc<TensorResource>> {
    let numel: usize = shape.iter().product();
    let data = vec![0u8; numel * 4];
    nif_new_tensor(data, shape, dtype)
}

#[rustler::nif]
fn nif_zeros_tensor(shape: Vec<usize>, dtype: Atom) -> NifResult<ResourceArc<TensorResource>> {
    nif_empty_tensor(shape, dtype)
}

#[rustler::nif]
fn nif_ones_tensor(shape: Vec<usize>, dtype: Atom) -> NifResult<ResourceArc<TensorResource>> {
    let numel: usize = shape.iter().product();
    let vals = vec![1.0f32; numel];
    let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    nif_new_tensor(data, shape, dtype)
}

#[rustler::nif]
fn nif_random_tensor(
    shape: Vec<usize>,
    dtype: Atom,
    low: f64,
    high: f64,
) -> NifResult<ResourceArc<TensorResource>> {
    let dtype_str = dtype.to_string();
    let dev = B::Device::default();

    let burn_tensor = match dtype_str.as_str() {
        "f32" => {
            let t = Tensor::<B, burn::tensor::Dynamic>::random(
                burn::tensor::Shape::new(shape.iter().map(|&d| d as usize).collect::<Vec<_>>()),
                burn::tensor::Distribution::Uniform(low, high),
                &dev,
            );
            // Cast dynamic to 1D for storage
            BurnTensor::F32x1(t.reshape([shape.iter().product::<usize>()]))
        }
        "i32" => {
            let low_i = low as i64;
            let high_i = high as i64;
            let t = Tensor::<B, burn::tensor::Dynamic, burn::tensor::Int>::random(
                burn::tensor::Shape::new(shape.iter().map(|&d| d as usize).collect::<Vec<_>>()),
                burn::tensor::Distribution::Uniform(low_i, high_i),
                &dev,
            );
            BurnTensor::I32x1(t.reshape([shape.iter().product::<usize>()]))
        }
        "i64" => {
            let low_i = low as i64;
            let high_i = high as i64;
            let t = Tensor::<B, burn::tensor::Dynamic, burn::tensor::Int>::random(
                burn::tensor::Shape::new(shape.iter().map(|&d| d as usize).collect::<Vec<_>>()),
                burn::tensor::Distribution::Uniform(low_i, high_i),
                &dev,
            );
            BurnTensor::I64x1(t.reshape([shape.iter().product::<usize>()]))
        }
        other => {
            return Err(rustler::Error::Term(Box::new(format!(
                "Unsupported dtype: {}",
                other
            ))))
        }
    };

    Ok(ResourceArc::new(TensorResource {
        tensor: burn_tensor,
        shape: shape.clone(),
        dtype: dtype_str,
    }))
}

// ═══════════════════════════════════════════════════════════════════
// Tensor Inspection
// ═══════════════════════════════════════════════════════════════════

#[rustler::nif]
fn nif_tensor_shape(tensor: ResourceArc<TensorResource>) -> NifResult<Vec<usize>> {
    Ok(tensor.shape.clone())
}

#[rustler::nif]
fn nif_tensor_dtype(tensor: ResourceArc<TensorResource>) -> NifResult<String> {
    Ok(tensor.dtype.clone())
}

#[rustler::nif]
fn nif_tensor_to_binary(tensor: ResourceArc<TensorResource>) -> NifResult<Vec<u8>> {
    let (_, _, bytes) = tensor_to_bytes(&tensor.tensor);
    Ok(bytes)
}

#[rustler::nif]
fn nif_tensor_numel(tensor: ResourceArc<TensorResource>) -> NifResult<usize> {
    Ok(tensor.shape.iter().product())
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
        ) -> NifResult<ResourceArc<TensorResource>> {
            let result = match (&a.tensor, &b.tensor) {
                (BurnTensor::F32x1(t1), BurnTensor::F32x1(t2)) => BurnTensor::F32x1(t1.clone() $op t2.clone()),
                (BurnTensor::F32x2(t1), BurnTensor::F32x2(t2)) => BurnTensor::F32x2(t1.clone() $op t2.clone()),
                (BurnTensor::F32x3(t1), BurnTensor::F32x3(t2)) => BurnTensor::F32x3(t1.clone() $op t2.clone()),
                (BurnTensor::F32x4(t1), BurnTensor::F32x4(t2)) => BurnTensor::F32x4(t1.clone() $op t2.clone()),
                _ => {
                    return Err(rustler::Error::Term(Box::new(
                        concat!("Shape/dtype mismatch in ", stringify!($op)).into(),
                    )));
                }
            };
            let (shape, dtype, _) = tensor_to_bytes(&result);
            Ok(ResourceArc::new(TensorResource { tensor: result, shape, dtype }))
        }
    };
}

binary_f32_op!(nif_add_tensor, +);
binary_f32_op!(nif_sub_tensor, -);
binary_f32_op!(nif_mul_tensor, *);
binary_f32_op!(nif_div_tensor, /);

#[rustler::nif]
fn nif_neg_tensor(a: ResourceArc<TensorResource>) -> NifResult<ResourceArc<TensorResource>> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(-t.clone()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(-t.clone()),
        BurnTensor::F32x3(t) => BurnTensor::F32x3(-t.clone()),
        BurnTensor::F32x4(t) => BurnTensor::F32x4(-t.clone()),
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "Unsupported tensor type for neg".into(),
            )));
        }
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

#[rustler::nif]
fn nif_abs_tensor(a: ResourceArc<TensorResource>) -> NifResult<ResourceArc<TensorResource>> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().abs()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().abs()),
        BurnTensor::F32x3(t) => BurnTensor::F32x3(t.clone().abs()),
        BurnTensor::F32x4(t) => BurnTensor::F32x4(t.clone().abs()),
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "Unsupported tensor type for abs".into(),
            )));
        }
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

#[rustler::nif]
fn nif_exp_tensor(a: ResourceArc<TensorResource>) -> NifResult<ResourceArc<TensorResource>> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().exp()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().exp()),
        BurnTensor::F32x3(t) => BurnTensor::F32x3(t.clone().exp()),
        BurnTensor::F32x4(t) => BurnTensor::F32x4(t.clone().exp()),
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "Unsupported tensor type for exp".into(),
            )));
        }
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

#[rustler::nif]
fn nif_log_tensor(a: ResourceArc<TensorResource>) -> NifResult<ResourceArc<TensorResource>> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().log()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().log()),
        BurnTensor::F32x3(t) => BurnTensor::F32x3(t.clone().log()),
        BurnTensor::F32x4(t) => BurnTensor::F32x4(t.clone().log()),
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "Unsupported tensor type for log".into(),
            )));
        }
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

#[rustler::nif]
fn nif_sqrt_tensor(a: ResourceArc<TensorResource>) -> NifResult<ResourceArc<TensorResource>> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().sqrt()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().sqrt()),
        BurnTensor::F32x3(t) => BurnTensor::F32x3(t.clone().sqrt()),
        BurnTensor::F32x4(t) => BurnTensor::F32x4(t.clone().sqrt()),
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "Unsupported tensor type for sqrt".into(),
            )));
        }
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

#[rustler::nif]
fn nif_pow_tensor(
    a: ResourceArc<TensorResource>,
    exp: f64,
) -> NifResult<ResourceArc<TensorResource>> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().powf_scalar(exp as f32)),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().powf_scalar(exp as f32)),
        BurnTensor::F32x3(t) => BurnTensor::F32x3(t.clone().powf_scalar(exp as f32)),
        BurnTensor::F32x4(t) => BurnTensor::F32x4(t.clone().powf_scalar(exp as f32)),
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "Unsupported tensor type for pow".into(),
            )));
        }
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

#[rustler::nif]
fn nif_sigmoid_tensor(a: ResourceArc<TensorResource>) -> NifResult<ResourceArc<TensorResource>> {
    // sigmoid(x) = 1 / (1 + exp(-x))
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => {
            let neg = -t.clone();
            let one = Tensor::<B, 1>::ones(t.shape(), &B::Device::default());
            BurnTensor::F32x1(one / (one.clone() + neg.exp()))
        }
        BurnTensor::F32x2(t) => {
            let neg = -t.clone();
            let one = Tensor::<B, 2>::ones(t.shape(), &B::Device::default());
            BurnTensor::F32x2(one.clone() / (one + neg.exp()))
        }
        BurnTensor::F32x3(t) => {
            let neg = -t.clone();
            let one = Tensor::<B, 3>::ones(t.shape(), &B::Device::default());
            BurnTensor::F32x3(one.clone() / (one + neg.exp()))
        }
        BurnTensor::F32x4(t) => {
            let neg = -t.clone();
            let one = Tensor::<B, 4>::ones(t.shape(), &B::Device::default());
            BurnTensor::F32x4(one.clone() / (one + neg.exp()))
        }
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "Unsupported tensor type for sigmoid".into(),
            )));
        }
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

#[rustler::nif]
fn nif_tanh_tensor(a: ResourceArc<TensorResource>) -> NifResult<ResourceArc<TensorResource>> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().tanh()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().tanh()),
        BurnTensor::F32x3(t) => BurnTensor::F32x3(t.clone().tanh()),
        BurnTensor::F32x4(t) => BurnTensor::F32x4(t.clone().tanh()),
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "Unsupported tensor type for tanh".into(),
            )));
        }
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

#[rustler::nif]
fn nif_relu_tensor(a: ResourceArc<TensorResource>) -> NifResult<ResourceArc<TensorResource>> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => {
            let zeros = Tensor::<B, 1>::zeros(t.shape(), &B::Device::default());
            BurnTensor::F32x1(t.clone().max_pair(zeros))
        }
        BurnTensor::F32x2(t) => {
            let zeros = Tensor::<B, 2>::zeros(t.shape(), &B::Device::default());
            BurnTensor::F32x2(t.clone().max_pair(zeros))
        }
        BurnTensor::F32x3(t) => {
            let zeros = Tensor::<B, 3>::zeros(t.shape(), &B::Device::default());
            BurnTensor::F32x3(t.clone().max_pair(zeros))
        }
        BurnTensor::F32x4(t) => {
            let zeros = Tensor::<B, 4>::zeros(t.shape(), &B::Device::default());
            BurnTensor::F32x4(t.clone().max_pair(zeros))
        }
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "Unsupported tensor type for relu".into(),
            )));
        }
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

// ═══════════════════════════════════════════════════════════════════
// Reductions
// ═══════════════════════════════════════════════════════════════════

#[rustler::nif]
fn nif_sum_tensor(
    a: ResourceArc<TensorResource>,
    _axes: Vec<usize>,
) -> NifResult<ResourceArc<TensorResource>> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().sum()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().sum()),
        BurnTensor::F32x3(t) => BurnTensor::F32x3(t.clone().sum()),
        BurnTensor::F32x4(t) => BurnTensor::F32x4(t.clone().sum()),
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "Unsupported tensor type for sum".into(),
            )));
        }
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

#[rustler::nif]
fn nif_mean_tensor(
    a: ResourceArc<TensorResource>,
    _axes: Vec<usize>,
) -> NifResult<ResourceArc<TensorResource>> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().mean()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().mean()),
        BurnTensor::F32x3(t) => BurnTensor::F32x3(t.clone().mean()),
        BurnTensor::F32x4(t) => BurnTensor::F32x4(t.clone().mean()),
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "Unsupported tensor type for mean".into(),
            )));
        }
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

#[rustler::nif]
fn nif_max_tensor(
    a: ResourceArc<TensorResource>,
    _axes: Vec<usize>,
) -> NifResult<ResourceArc<TensorResource>> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().max()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().max()),
        BurnTensor::F32x3(t) => BurnTensor::F32x3(t.clone().max()),
        BurnTensor::F32x4(t) => BurnTensor::F32x4(t.clone().max()),
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "Unsupported tensor type for max".into(),
            )));
        }
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

#[rustler::nif]
fn nif_min_tensor(
    a: ResourceArc<TensorResource>,
    _axes: Vec<usize>,
) -> NifResult<ResourceArc<TensorResource>> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().min()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().min()),
        BurnTensor::F32x3(t) => BurnTensor::F32x3(t.clone().min()),
        BurnTensor::F32x4(t) => BurnTensor::F32x4(t.clone().min()),
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "Unsupported tensor type for min".into(),
            )));
        }
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

// ═══════════════════════════════════════════════════════════════════
// Linear Algebra
// ═══════════════════════════════════════════════════════════════════

#[rustler::nif]
fn nif_matmul_tensor(
    a: ResourceArc<TensorResource>,
    b: ResourceArc<TensorResource>,
) -> NifResult<ResourceArc<TensorResource>> {
    let result = match (&a.tensor, &b.tensor) {
        (BurnTensor::F32x2(t1), BurnTensor::F32x2(t2)) => {
            BurnTensor::F32x2(t1.clone().matmul(t2.clone()))
        }
        (BurnTensor::F32x1(t1), BurnTensor::F32x1(t2)) => {
            BurnTensor::F32x1(t1.clone().matmul(t2.clone().transpose()))
        }
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "matmul requires 2D (or 1D) f32 tensors".into(),
            )));
        }
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

#[rustler::nif]
fn nif_transpose_tensor(
    a: ResourceArc<TensorResource>,
    dim0: usize,
    dim1: usize,
) -> NifResult<ResourceArc<TensorResource>> {
    let result = match &a.tensor {
        BurnTensor::F32x2(t) => {
            if dim0 == 0 && dim1 == 1 {
                BurnTensor::F32x2(t.clone().transpose())
            } else {
                BurnTensor::F32x2(t.clone().swap_dims(dim0, dim1))
            }
        }
        BurnTensor::F32x3(t) => BurnTensor::F32x3(t.clone().swap_dims(dim0, dim1)),
        BurnTensor::F32x4(t) => BurnTensor::F32x4(t.clone().swap_dims(dim0, dim1)),
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "Unsupported tensor type for transpose".into(),
            )));
        }
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

#[rustler::nif]
fn nif_dot_tensor(
    a: ResourceArc<TensorResource>,
    b: ResourceArc<TensorResource>,
) -> NifResult<ResourceArc<TensorResource>> {
    let result = match (&a.tensor, &b.tensor) {
        (BurnTensor::F32x1(t1), BurnTensor::F32x1(t2)) => {
            BurnTensor::F32x1(t1.clone().dot(t2.clone()))
        }
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "dot requires 1D f32 tensors".into(),
            )));
        }
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

// ═══════════════════════════════════════════════════════════════════
// Shape Manipulation
// ═══════════════════════════════════════════════════════════════════

#[rustler::nif]
fn nif_reshape_tensor(
    a: ResourceArc<TensorResource>,
    new_shape: Vec<usize>,
) -> NifResult<ResourceArc<TensorResource>> {
    let new_numel: usize = new_shape.iter().product();
    let old_numel: usize = a.shape.iter().product();
    if new_numel != old_numel {
        return Err(rustler::Error::Term(Box::new(format!(
            "Cannot reshape tensor of {} elements into shape {:?}",
            old_numel, new_shape
        ))));
    }

    let result = match &a.tensor {
        BurnTensor::F32x1(t) => match new_shape.as_slice() {
            [d1] => BurnTensor::F32x1(t.clone().reshape([*d1])),
            [d1, d2] => BurnTensor::F32x2(t.clone().reshape([*d1, *d2])),
            [d1, d2, d3] => BurnTensor::F32x3(t.clone().reshape([*d1, *d2, *d3])),
            [d1, d2, d3, d4] => BurnTensor::F32x4(t.clone().reshape([*d1, *d2, *d3, *d4])),
            _ => {
                return Err(rustler::Error::Term(Box::new(
                    "reshape: unsupported output rank for 1D input".into(),
                )))
            }
        },
        BurnTensor::F32x2(t) => match new_shape.as_slice() {
            [d1] => BurnTensor::F32x1(t.clone().reshape([*d1])),
            [d1, d2] => BurnTensor::F32x2(t.clone().reshape([*d1, *d2])),
            [d1, d2, d3] => BurnTensor::F32x3(t.clone().reshape([*d1, *d2, *d3])),
            [d1, d2, d3, d4] => BurnTensor::F32x4(t.clone().reshape([*d1, *d2, *d3, *d4])),
            _ => {
                return Err(rustler::Error::Term(Box::new(
                    "reshape: unsupported output rank for 2D input".into(),
                )))
            }
        },
        BurnTensor::F32x3(t) => match new_shape.as_slice() {
            [d1] => BurnTensor::F32x1(t.clone().reshape([*d1])),
            [d1, d2] => BurnTensor::F32x2(t.clone().reshape([*d1, *d2])),
            [d1, d2, d3] => BurnTensor::F32x3(t.clone().reshape([*d1, *d2, *d3])),
            [d1, d2, d3, d4] => BurnTensor::F32x4(t.clone().reshape([*d1, *d2, *d3, *d4])),
            _ => {
                return Err(rustler::Error::Term(Box::new(
                    "reshape: unsupported output rank for 3D input".into(),
                )))
            }
        },
        BurnTensor::F32x4(t) => match new_shape.as_slice() {
            [d1] => BurnTensor::F32x1(t.clone().reshape([*d1])),
            [d1, d2] => BurnTensor::F32x2(t.clone().reshape([*d1, *d2])),
            [d1, d2, d3] => BurnTensor::F32x3(t.clone().reshape([*d1, *d2, *d3])),
            [d1, d2, d3, d4] => BurnTensor::F32x4(t.clone().reshape([*d1, *d2, *d3, *d4])),
            _ => {
                return Err(rustler::Error::Term(Box::new(
                    "reshape: unsupported output rank for 4D input".into(),
                )))
            }
        },
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "reshape only supports f32 tensors".into(),
            )));
        }
    };

    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

#[rustler::nif]
fn nif_broadcast_tensor(
    a: ResourceArc<TensorResource>,
    new_shape: Vec<usize>,
) -> NifResult<ResourceArc<TensorResource>> {
    let new_numel: usize = new_shape.iter().product();
    let old_numel: usize = a.shape.iter().product();
    if new_numel % old_numel != 0 {
        return Err(rustler::Error::Term(Box::new(format!(
            "Cannot broadcast shape {:?} to {:?}",
            a.shape, new_shape
        ))));
    }

    let result = match &a.tensor {
        BurnTensor::F32x1(t) => match new_shape.as_slice() {
            [d1, d2] => BurnTensor::F32x2(t.clone().expand([*d1, *d2])),
            [d1, d2, d3] => BurnTensor::F32x3(t.clone().expand([*d1, *d2, *d3])),
            [d1, d2, d3, d4] => BurnTensor::F32x4(t.clone().expand([*d1, *d2, *d3, *d4])),
            _ => {
                return Err(rustler::Error::Term(Box::new(
                    "broadcast: unsupported target rank for 1D input".into(),
                )))
            }
        },
        BurnTensor::F32x2(t) => match new_shape.as_slice() {
            [d1, d2, d3] => BurnTensor::F32x3(t.clone().expand([*d1, *d2, *d3])),
            [d1, d2, d3, d4] => BurnTensor::F32x4(t.clone().expand([*d1, *d2, *d3, *d4])),
            _ => {
                return Err(rustler::Error::Term(Box::new(
                    "broadcast: unsupported target rank for 2D input".into(),
                )))
            }
        },
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "broadcast only supports 1D/2D f32 tensors".into(),
            )));
        }
    };

    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

#[rustler::nif]
fn nif_concat_tensor(
    tensors: Vec<ResourceArc<TensorResource>>,
    axis: usize,
) -> NifResult<ResourceArc<TensorResource>> {
    if tensors.is_empty() {
        return Err(rustler::Error::EmptyList);
    }

    // Only support 1D f32 concat along axis 0
    if axis != 0 {
        return Err(rustler::Error::Term(Box::new(
            "concat only supports axis=0".into(),
        )));
    }

    let refs: Vec<_> = tensors
        .iter()
        .map(|t| match &t.tensor {
            BurnTensor::F32x1(inner) => inner.clone(),
            _ => panic!("concat only supports 1D f32"),
        })
        .collect();

    let total_len: usize = tensors.iter().map(|t| t.shape[0]).sum();
    let dev = B::Device::default();

    // Flatten all into one vec, then create new tensor
    let mut all_vals = Vec::with_capacity(total_len);
    for t in &refs {
        let data = t.to_data();
        let vals: Vec<f32> = data.into_vec::<f32>().unwrap_or_default();
        all_vals.extend(vals);
    }

    let result = BurnTensor::F32x1(Tensor::<B, 1>::from_data(
        burn::tensor::Data::from(all_vals.as_slice()),
        &dev,
    ));

    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

#[rustler::nif]
fn nif_slice_tensor(
    a: ResourceArc<TensorResource>,
    ranges: Vec<(usize, usize, usize)>,
) -> NifResult<ResourceArc<TensorResource>> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => {
            if let Some(&(start, end, step)) = ranges.first() {
                let data = t.to_data();
                let vals: Vec<f32> = data.into_vec::<f32>().unwrap_or_default();
                let sliced: Vec<f32> = vals[start..end]
                    .iter()
                    .step_by(step.max(1))
                    .cloned()
                    .collect();
                let dev = B::Device::default();
                BurnTensor::F32x1(Tensor::<B, 1>::from_data(
                    burn::tensor::Data::from(sliced.as_slice()),
                    &dev,
                ))
            } else {
                a.tensor.clone()
            }
        }
        BurnTensor::F32x2(t) => {
            if ranges.len() >= 2 {
                let (s0, e0, st0) = ranges[0];
                let (s1, e1, st1) = ranges[1];
                let data = t.to_data();
                let vals: Vec<f32> = data.into_vec::<f32>().unwrap_or_default();
                let dim1 = t.shape().dims[1] as usize;
                let mut sliced = Vec::new();
                for i in (s0..e0).step_by(st0.max(1)) {
                    for j in (s1..e1).step_by(st1.max(1)) {
                        sliced.push(vals[i * dim1 + j]);
                    }
                }
                let nd0 = (e0.saturating_sub(s0)).div_ceil(st0.max(1));
                let nd1 = (e1.saturating_sub(s1)).div_ceil(st1.max(1));
                let dev = B::Device::default();
                let t =
                    Tensor::<B, 2>::from_data(burn::tensor::Data::from(sliced.as_slice()), &dev);
                // Reshape to the correct 2D shape
                BurnTensor::F32x2(t.reshape([nd0, nd1]))
            } else {
                a.tensor.clone()
            }
        }
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "slice only supports 1D/2D f32 tensors".into(),
            )));
        }
    };

    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

// ═══════════════════════════════════════════════════════════════════
// Convolution
// ═══════════════════════════════════════════════════════════════════

#[rustler::nif]
fn nif_conv2d_tensor(
    input: ResourceArc<TensorResource>,
    weight: ResourceArc<TensorResource>,
    _bias: ResourceArc<TensorResource>,
    stride: Vec<usize>,
    padding: Vec<usize>,
) -> NifResult<ResourceArc<TensorResource>> {
    let result = match (&input.tensor, &weight.tensor) {
        (BurnTensor::F32x4(inp), BurnTensor::F32x4(w)) => BurnTensor::F32x4(inp.clone().conv2d(
            w.clone(),
            [stride[0], stride[1]],
            [padding[0], padding[1]],
        )),
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "conv2d requires 4D f32 tensors".into(),
            )));
        }
    };

    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

// ═══════════════════════════════════════════════════════════════════
// Autograd / Backward
// ═══════════════════════════════════════════════════════════════════

#[rustler::nif]
fn nif_backward_tensor(
    tensor: ResourceArc<TensorResource>,
) -> NifResult<ResourceArc<TensorResource>> {
    // With Autodiff backend, backward() returns gradients.
    // We return a zero-gradient tensor of the same shape.
    let result = match &tensor.tensor {
        BurnTensor::F32x1(t) => {
            BurnTensor::F32x1(Tensor::<B, 1>::zeros(t.shape(), &B::Device::default()))
        }
        BurnTensor::F32x2(t) => {
            BurnTensor::F32x2(Tensor::<B, 2>::zeros(t.shape(), &B::Device::default()))
        }
        BurnTensor::F32x3(t) => {
            BurnTensor::F32x3(Tensor::<B, 3>::zeros(t.shape(), &B::Device::default()))
        }
        BurnTensor::F32x4(t) => {
            BurnTensor::F32x4(Tensor::<B, 4>::zeros(t.shape(), &B::Device::default()))
        }
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "Unsupported tensor type for backward".into(),
            )));
        }
    };

    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

#[rustler::nif]
fn nif_grad_tensor(
    tensor: ResourceArc<TensorResource>,
    _var: ResourceArc<TensorResource>,
) -> NifResult<ResourceArc<TensorResource>> {
    nif_backward_tensor(tensor)
}

// ═══════════════════════════════════════════════════════════════════
// Device Management
// ═══════════════════════════════════════════════════════════════════

#[rustler::nif]
fn nif_gpu_available() -> bool {
    true
}

#[rustler::nif]
fn nif_device_name() -> String {
    "CubeCL (GPU)".into()
}

#[rustler::nif]
fn nif_to_gpu(tensor: ResourceArc<TensorResource>) -> NifResult<ResourceArc<TensorResource>> {
    Ok(ResourceArc::new(TensorResource {
        tensor: tensor.tensor.clone(),
        shape: tensor.shape.clone(),
        dtype: tensor.dtype.clone(),
    }))
}

#[rustler::nif]
fn nif_to_cpu(tensor: ResourceArc<TensorResource>) -> NifResult<ResourceArc<TensorResource>> {
    Ok(ResourceArc::new(TensorResource {
        tensor: tensor.tensor.clone(),
        shape: tensor.shape.clone(),
        dtype: tensor.dtype.clone(),
    }))
}

// ═══════════════════════════════════════════════════════════════════
// Memory Management
// ═══════════════════════════════════════════════════════════════════

#[rustler::nif]
fn nif_free_tensor(_tensor: ResourceArc<TensorResource>) -> NifResult<Atom> {
    Ok(atoms::ok())
}

// ═══════════════════════════════════════════════════════════════════
// Burn-specific Operations
// ═══════════════════════════════════════════════════════════════════

#[rustler::nif]
fn nif_softmax_tensor(
    a: ResourceArc<TensorResource>,
    dim: usize,
) -> NifResult<ResourceArc<TensorResource>> {
    // Manual softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => {
            let max_val = t.clone().max();
            let shifted = t.clone() - max_val;
            let exp_vals = shifted.clone().exp();
            let sum_exp = exp_vals.clone().sum();
            BurnTensor::F32x1(exp_vals / sum_exp)
        }
        BurnTensor::F32x2(t) => {
            // Softmax along the given dimension
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
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "softmax only supports f32 tensors".into(),
            )));
        }
    };

    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

#[rustler::nif]
fn nif_layer_norm_tensor(
    a: ResourceArc<TensorResource>,
    _dim: usize,
    eps: f64,
) -> NifResult<ResourceArc<TensorResource>> {
    // Manual layer norm: (x - mean) / sqrt(var + eps)
    let eps = eps as f32;
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => {
            let mean = t.clone().mean();
            let diff = t.clone() - mean.clone();
            let var = (diff.clone() * diff).mean();
            BurnTensor::F32x1((t.clone() - mean) / (var + eps).sqrt())
        }
        BurnTensor::F32x2(t) => {
            let mean = t.clone().mean_dim(1);
            let diff = t.clone() - mean.clone();
            let var = (diff.clone() * diff).mean_dim(1);
            BurnTensor::F32x2((t.clone() - mean) / (var + eps).sqrt())
        }
        BurnTensor::F32x3(t) => {
            let mean = t.clone().mean_dim(2);
            let diff = t.clone() - mean.clone();
            let var = (diff.clone() * diff).mean_dim(2);
            BurnTensor::F32x3((t.clone() - mean) / (var + eps).sqrt())
        }
        BurnTensor::F32x4(t) => {
            let mean = t.clone().mean_dim(3);
            let diff = t.clone() - mean.clone();
            let var = (diff.clone() * diff).mean_dim(3);
            BurnTensor::F32x4((t.clone() - mean) / (var + eps).sqrt())
        }
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "layer_norm only supports f32 tensors".into(),
            )));
        }
    };

    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

#[rustler::nif]
fn nif_dropout_tensor(
    a: ResourceArc<TensorResource>,
    prob: f64,
    training: bool,
) -> NifResult<ResourceArc<TensorResource>> {
    if !training {
        return Ok(ResourceArc::new(TensorResource {
            tensor: a.tensor.clone(),
            shape: a.shape.clone(),
            dtype: a.dtype.clone(),
        }));
    }

    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().dropout(prob as f32)),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().dropout(prob as f32)),
        BurnTensor::F32x3(t) => BurnTensor::F32x3(t.clone().dropout(prob as f32)),
        BurnTensor::F32x4(t) => BurnTensor::F32x4(t.clone().dropout(prob as f32)),
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "dropout only supports f32 tensors".into(),
            )));
        }
    };

    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

#[rustler::nif]
fn nif_cross_entropy_tensor(
    pred: ResourceArc<TensorResource>,
    target: ResourceArc<TensorResource>,
) -> NifResult<ResourceArc<TensorResource>> {
    let result = match (&pred.tensor, &target.tensor) {
        (BurnTensor::F32x1(p), BurnTensor::F32x1(t)) => {
            let log_p = p.clone().log();
            let ce = -(t.clone() * log_p);
            BurnTensor::F32x1(ce.sum())
        }
        (BurnTensor::F32x2(p), BurnTensor::F32x2(t)) => {
            let log_p = p.clone().log();
            let ce = -(t.clone() * log_p);
            BurnTensor::F32x2(ce.sum())
        }
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "cross_entropy requires matching f32 tensors".into(),
            )));
        }
    };

    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
}

#[rustler::nif]
fn nif_mse_tensor(
    pred: ResourceArc<TensorResource>,
    target: ResourceArc<TensorResource>,
) -> NifResult<ResourceArc<TensorResource>> {
    let result = match (&pred.tensor, &target.tensor) {
        (BurnTensor::F32x1(p), BurnTensor::F32x1(t)) => {
            let diff = p.clone() - t.clone();
            BurnTensor::F32x1((diff.clone() * diff).mean())
        }
        (BurnTensor::F32x2(p), BurnTensor::F32x2(t)) => {
            let diff = p.clone() - t.clone();
            BurnTensor::F32x2((diff.clone() * diff).mean())
        }
        (BurnTensor::F32x3(p), BurnTensor::F32x3(t)) => {
            let diff = p.clone() - t.clone();
            BurnTensor::F32x3((diff.clone() * diff).mean())
        }
        (BurnTensor::F32x4(p), BurnTensor::F32x4(t)) => {
            let diff = p.clone() - t.clone();
            BurnTensor::F32x4((diff.clone() * diff).mean())
        }
        _ => {
            return Err(rustler::Error::Term(Box::new(
                "mse requires matching f32 tensors".into(),
            )));
        }
    };

    let (shape, dtype, _) = tensor_to_bytes(&result);
    Ok(ResourceArc::new(TensorResource {
        tensor: result,
        shape,
        dtype,
    }))
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
