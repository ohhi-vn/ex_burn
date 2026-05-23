//! # ExBurn NIF

use rustler::ResourceArc;
use std::panic::{RefUnwindSafe, UnwindSafe};

use burn::tensor::Tensor;
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;

type B = Autodiff<NdArray>;

fn device() -> burn_ndarray::NdArrayDevice {
    burn_ndarray::NdArrayDevice::default()
}

#[derive(Clone)]
pub enum BurnTensor {
    F32x1(Tensor<B, 1>),
    F32x2(Tensor<B, 2>),
    F32x3(Tensor<B, 3>),
    F32x4(Tensor<B, 4>),
}

pub struct TensorResource {
    pub tensor: BurnTensor,
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[rustler::resource_impl]
impl rustler::Resource for TensorResource {}
impl RefUnwindSafe for TensorResource {}
impl UnwindSafe for TensorResource {}

rustler::init!("Elixir.ExBurn.Nif");

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

fn tensor_to_bytes(t: &BurnTensor) -> (Vec<usize>, String, Vec<u8>) {
    match t {
        BurnTensor::F32x1(t) => {
            let dims: Vec<usize> = t.shape().dims::<1>().to_vec();
            let vals: Vec<f32> = t.to_data().into_vec().unwrap_or_default();
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            (dims, "f32".into(), bytes)
        }
        BurnTensor::F32x2(t) => {
            let dims: Vec<usize> = t.shape().dims::<2>().to_vec();
            let vals: Vec<f32> = t.to_data().into_vec().unwrap_or_default();
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            (dims, "f32".into(), bytes)
        }
        BurnTensor::F32x3(t) => {
            let dims: Vec<usize> = t.shape().dims::<3>().to_vec();
            let vals: Vec<f32> = t.to_data().into_vec().unwrap_or_default();
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            (dims, "f32".into(), bytes)
        }
        BurnTensor::F32x4(t) => {
            let dims: Vec<usize> = t.shape().dims::<4>().to_vec();
            let vals: Vec<f32> = t.to_data().into_vec().unwrap_or_default();
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            (dims, "f32".into(), bytes)
        }
    }
}

fn make_f32_tensor(vals: &[f32], shape: &[usize]) -> BurnTensor {
    let dev = device();
    match shape.len() {
        1 => BurnTensor::F32x1(Tensor::<B, 1>::from_floats(vals, &dev)),
        2 => BurnTensor::F32x2(Tensor::<B, 2>::from_floats(vals, &dev)),
        3 => BurnTensor::F32x3(Tensor::<B, 3>::from_floats(vals, &dev)),
        4 => BurnTensor::F32x4(Tensor::<B, 4>::from_floats(vals, &dev)),
        _ => panic!("Unsupported rank {}", shape.len()),
    }
}

fn make_tensor_from_bytes(data: Vec<u8>, shape: Vec<usize>, dtype: String) -> BurnTensor {
    match dtype.as_str() {
        "f32" => {
            let vals: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            make_f32_tensor(&vals, &shape)
        }
        other => panic!("Unsupported dtype: {}", other),
    }
}

fn build_resource(t: BurnTensor, shape: Vec<usize>, dtype: String) -> ResourceArc<TensorResource> {
    ResourceArc::new(TensorResource {
        tensor: t,
        shape,
        dtype,
    })
}

// ═══════════════════════════════════════════════════════════════════
// Tensor Creation
// ═══════════════════════════════════════════════════════════════════

#[rustler::nif]
fn nif_new_tensor(data: Vec<u8>, shape: Vec<usize>, dtype: String) -> ResourceArc<TensorResource> {
    let t = make_tensor_from_bytes(data, shape.clone(), dtype.clone());
    build_resource(t, shape, dtype)
}

#[rustler::nif]
fn nif_empty_tensor(shape: Vec<usize>, dtype: String) -> ResourceArc<TensorResource> {
    let numel: usize = shape.iter().product();
    let data = vec![0u8; numel * 4];
    let t = make_tensor_from_bytes(data, shape.clone(), dtype.clone());
    build_resource(t, shape, dtype)
}

#[rustler::nif]
fn nif_zeros_tensor(shape: Vec<usize>, dtype: String) -> ResourceArc<TensorResource> {
    let numel: usize = shape.iter().product();
    let data = vec![0u8; numel * 4];
    let t = make_tensor_from_bytes(data, shape.clone(), dtype.clone());
    build_resource(t, shape, dtype)
}

#[rustler::nif]
fn nif_ones_tensor(shape: Vec<usize>, dtype: String) -> ResourceArc<TensorResource> {
    let numel: usize = shape.iter().product();
    let vals = vec![1.0f32; numel];
    let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let t = make_tensor_from_bytes(data, shape.clone(), dtype.clone());
    build_resource(t, shape, dtype)
}

#[rustler::nif]
fn nif_eye_tensor(size: usize, _type: String) -> ResourceArc<TensorResource> {
    let dev = device();
    let t = Tensor::<B, 2>::eye(size, &dev);
    build_resource(BurnTensor::F32x2(t), vec![size, size], "f32".into())
}

#[rustler::nif]
fn nif_iota_tensor(shape: Vec<usize>, axis: usize, _type: String) -> ResourceArc<TensorResource> {
    let dev = device();
    let n = shape.get(axis).copied().unwrap_or(1);
    let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let t = Tensor::<B, 1>::from_floats(vals.as_slice(), &dev);
    build_resource(BurnTensor::F32x1(t), vec![n], "f32".into())
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
// Arithmetic
// ═══════════════════════════════════════════════════════════════════

#[rustler::nif]
fn nif_add_tensor(
    a: ResourceArc<TensorResource>,
    b: ResourceArc<TensorResource>,
) -> ResourceArc<TensorResource> {
    let result = match (&a.tensor, &b.tensor) {
        (BurnTensor::F32x1(t1), BurnTensor::F32x1(t2)) => {
            BurnTensor::F32x1(t1.clone() + t2.clone())
        }
        (BurnTensor::F32x2(t1), BurnTensor::F32x2(t2)) => {
            BurnTensor::F32x2(t1.clone() + t2.clone())
        }
        _ => panic!("Shape/dtype mismatch in add"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[rustler::nif]
fn nif_sub_tensor(
    a: ResourceArc<TensorResource>,
    b: ResourceArc<TensorResource>,
) -> ResourceArc<TensorResource> {
    let result = match (&a.tensor, &b.tensor) {
        (BurnTensor::F32x1(t1), BurnTensor::F32x1(t2)) => {
            BurnTensor::F32x1(t1.clone() - t2.clone())
        }
        (BurnTensor::F32x2(t1), BurnTensor::F32x2(t2)) => {
            BurnTensor::F32x2(t1.clone() - t2.clone())
        }
        _ => panic!("Shape/dtype mismatch in sub"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[rustler::nif]
fn nif_mul_tensor(
    a: ResourceArc<TensorResource>,
    b: ResourceArc<TensorResource>,
) -> ResourceArc<TensorResource> {
    let result = match (&a.tensor, &b.tensor) {
        (BurnTensor::F32x1(t1), BurnTensor::F32x1(t2)) => {
            BurnTensor::F32x1(t1.clone() * t2.clone())
        }
        (BurnTensor::F32x2(t1), BurnTensor::F32x2(t2)) => {
            BurnTensor::F32x2(t1.clone() * t2.clone())
        }
        _ => panic!("Shape/dtype mismatch in mul"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[rustler::nif]
fn nif_div_tensor(
    a: ResourceArc<TensorResource>,
    b: ResourceArc<TensorResource>,
) -> ResourceArc<TensorResource> {
    let result = match (&a.tensor, &b.tensor) {
        (BurnTensor::F32x1(t1), BurnTensor::F32x1(t2)) => {
            BurnTensor::F32x1(t1.clone() / t2.clone())
        }
        (BurnTensor::F32x2(t1), BurnTensor::F32x2(t2)) => {
            BurnTensor::F32x2(t1.clone() / t2.clone())
        }
        _ => panic!("Shape/dtype mismatch in div"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[rustler::nif]
fn nif_neg_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(-t.clone()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(-t.clone()),
        _ => panic!("Unsupported tensor type for neg"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[rustler::nif]
fn nif_abs_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().abs()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().abs()),
        _ => panic!("Unsupported tensor type for abs"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[rustler::nif]
fn nif_exp_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().exp()),
        _ => panic!("Unsupported tensor type for exp"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[rustler::nif]
fn nif_log_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().log()),
        _ => panic!("Unsupported tensor type for log"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[rustler::nif]
fn nif_sqrt_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().sqrt()),
        _ => panic!("Unsupported tensor type for sqrt"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[rustler::nif]
fn nif_pow_tensor(a: ResourceArc<TensorResource>, exp: f32) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().powf(Tensor::<B, 1>::full(
            t.shape(),
            exp,
            &device(),
        ))),
        _ => panic!("Unsupported tensor type for pow"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[rustler::nif]
fn nif_sigmoid_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => {
            let one = Tensor::<B, 1>::ones(t.shape(), &device());
            BurnTensor::F32x1(one.clone() / (one + (-t.clone()).exp()))
        }
        _ => panic!("Unsupported tensor type for sigmoid"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[rustler::nif]
fn nif_tanh_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().tanh()),
        _ => panic!("Unsupported tensor type for tanh"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[rustler::nif]
fn nif_relu_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => {
            let z = Tensor::<B, 1>::zeros(t.shape(), &device());
            BurnTensor::F32x1(t.clone().max_pair(z))
        }
        _ => panic!("Unsupported tensor type for relu"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

// ═══════════════════════════════════════════════════════════════════
// Reductions
// ═══════════════════════════════════════════════════════════════════

#[rustler::nif]
fn nif_sum_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().sum()),
        BurnTensor::F32x2(t) => BurnTensor::F32x1(t.clone().sum()),
        _ => panic!("Unsupported tensor type for sum"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[rustler::nif]
fn nif_mean_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().mean()),
        _ => panic!("Unsupported tensor type for mean"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[rustler::nif]
fn nif_max_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().max()),
        _ => panic!("Unsupported tensor type for max"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[rustler::nif]
fn nif_min_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().min()),
        _ => panic!("Unsupported tensor type for min"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

// ═══════════════════════════════════════════════════════════════════
// Linear Algebra
// ═══════════════════════════════════════════════════════════════════

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
    build_resource(result, shape, dtype)
}

#[rustler::nif]
fn nif_transpose_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().transpose()),
        _ => panic!("Transpose requires 2D tensor"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
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
    build_resource(result, shape, dtype)
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
        panic!("Cannot reshape {} elements into {:?}", old_numel, new_shape);
    }
    let result = match (&a.tensor, new_shape.as_slice()) {
        (BurnTensor::F32x1(t), [d1]) => BurnTensor::F32x1(t.clone().reshape([*d1])),
        (BurnTensor::F32x1(t), [d1, d2]) => BurnTensor::F32x2(t.clone().reshape([*d1, *d2])),
        (BurnTensor::F32x1(t), [d1, d2, d3]) => {
            BurnTensor::F32x3(t.clone().reshape([*d1, *d2, *d3]))
        }
        (BurnTensor::F32x1(t), [d1, d2, d3, d4]) => {
            BurnTensor::F32x4(t.clone().reshape([*d1, *d2, *d3, *d4]))
        }
        (BurnTensor::F32x2(t), [d1]) => BurnTensor::F32x1(t.clone().reshape([*d1])),
        (BurnTensor::F32x2(t), [d1, d2]) => BurnTensor::F32x2(t.clone().reshape([*d1, *d2])),
        _ => panic!("reshape: unsupported"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[rustler::nif]
fn nif_broadcast_tensor(
    a: ResourceArc<TensorResource>,
    target_shape: Vec<usize>,
) -> ResourceArc<TensorResource> {
    let result = match (&a.tensor, target_shape.as_slice()) {
        (BurnTensor::F32x1(t), [d1]) => BurnTensor::F32x1(t.clone().reshape([*d1])),
        (BurnTensor::F32x1(t), [d1, d2]) => BurnTensor::F32x2(t.clone().reshape([*d1, *d2])),
        (BurnTensor::F32x2(t), [d1, d2]) => BurnTensor::F32x2(t.clone().reshape([*d1, *d2])),
        _ => panic!("broadcast: unsupported"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[rustler::nif]
fn nif_concat_tensor(
    a: ResourceArc<TensorResource>,
    b: ResourceArc<TensorResource>,
) -> ResourceArc<TensorResource> {
    let total_len = a.shape[0] + b.shape[0];
    let dev = device();
    let mut all_vals: Vec<f32> = Vec::with_capacity(total_len);
    if let BurnTensor::F32x1(inner) = &a.tensor {
        let vals: Vec<f32> = inner.to_data().into_vec().unwrap_or_default();
        all_vals.extend(vals);
    }
    if let BurnTensor::F32x1(inner) = &b.tensor {
        let vals: Vec<f32> = inner.to_data().into_vec().unwrap_or_default();
        all_vals.extend(vals);
    }
    let result = BurnTensor::F32x1(Tensor::<B, 1>::from_floats(all_vals.as_slice(), &dev));
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

// ═══════════════════════════════════════════════════════════════════
// Device & Memory
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
    build_resource(
        tensor.tensor.clone(),
        tensor.shape.clone(),
        tensor.dtype.clone(),
    )
}

#[rustler::nif]
fn nif_to_cpu(tensor: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    build_resource(
        tensor.tensor.clone(),
        tensor.shape.clone(),
        tensor.dtype.clone(),
    )
}

#[rustler::nif]
fn nif_free_tensor(_tensor: ResourceArc<TensorResource>) -> rustler::Atom {
    rustler::types::atom::ok()
}

// ═══════════════════════════════════════════════════════════════════
// Neural Network
// ═══════════════════════════════════════════════════════════════════

#[rustler::nif]
fn nif_softmax_tensor(a: ResourceArc<TensorResource>, dim: i64) -> ResourceArc<TensorResource> {
    let dim = if dim < 0 {
        a.shape.len() as usize
    } else {
        dim as usize
    };
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
        _ => panic!("softmax only supports f32 tensors"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[rustler::nif]
fn nif_layer_norm_tensor(
    a: ResourceArc<TensorResource>,
    _dim: i64,
    _eps: f64,
) -> ResourceArc<TensorResource> {
    build_resource(a.tensor.clone(), a.shape.clone(), a.dtype.clone())
}
