//! # ExBurn NIF
//!
//! Rust NIF bridge between Elixir and the Burn deep learning framework.
//! Supports multiple backends:
//!   - **CUDA** (NVIDIA GPUs) — via `burn/cuda` feature
//!   - **Metal** (Apple GPUs) — via `burn/metal` feature
//!   - **Vulkan** (Android/Linux/Windows) — via `burn/vulkan` feature
//!   - **NdArray** (CPU fallback) — always available
//!
//! The backend is selected at compile time via Cargo features on the `burn` crate.
//! The default feature is `cuda`.  Build with `--features metal` or
//! `--features vulkan` to target other GPUs, or `--no-default-features`
//! for CPU-only (NdArray).

use rustler::ResourceArc;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::OnceLock;

use burn::tensor::Tensor;
use burn_autodiff::Autodiff;
#[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
use burn_ndarray::NdArray;

// ── Backend selection ─────────────────────────────────────────────
//
// The active GPU backend is chosen at compile time via Cargo features.
// Each feature gate brings in the corresponding Burn backend and device
// types.  When no GPU feature is enabled we fall back to NdArray (CPU).

/// The concrete backend type used throughout this NIF.
///
/// With the `cuda` feature this is a CUDA GPU backend.
/// With `metal` it is a Metal GPU backend.
/// With `vulkan` it is a Vulkan GPU backend.
/// With no GPU feature it falls back to NdArray (CPU).
#[cfg(feature = "cuda")]
type GpuBackend = burn::backend::Cuda;

#[cfg(feature = "metal")]
type GpuBackend = burn::backend::Metal;

#[cfg(feature = "vulkan")]
type GpuBackend = burn::backend::Vulkan;

#[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
type GpuBackend = NdArray;

/// Autodiff wrapper around the GPU backend.
#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
type B = Autodiff<GpuBackend>;

/// Autodiff wrapper around NdArray (CPU fallback).
#[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
type B = Autodiff<NdArray>;

// ── Device ────────────────────────────────────────────────────────
// The `BackendDevice` trait abstracts over the concrete device type
// for each backend.  Each backend cfg provides its own impl so the
// rest of the code can call `device()` uniformly.

/// Trait to obtain the default device for the active backend.
trait BackendDevice {
    type Device;
    fn device() -> Self::Device;
}

#[cfg(feature = "cuda")]
impl BackendDevice for GpuBackend {
    type Device = burn::backend::cuda::CudaDevice;
    fn device() -> Self::Device {
        burn::backend::cuda::CudaDevice::default()
    }
}

#[cfg(feature = "metal")]
impl BackendDevice for GpuBackend {
    type Device = burn::backend::metal::MetalDevice;
    fn device() -> Self::Device {
        burn::backend::metal::MetalDevice::default()
    }
}

#[cfg(feature = "vulkan")]
impl BackendDevice for GpuBackend {
    type Device = burn::backend::vulkan::VulkanDevice;
    fn device() -> Self::Device {
        burn::backend::vulkan::VulkanDevice::default()
    }
}

#[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
impl BackendDevice for NdArray {
    type Device = burn_ndarray::NdArrayDevice;
    fn device() -> Self::Device {
        burn_ndarray::NdArrayDevice::default()
    }
}

/// Returns the default device for the compiled backend.
fn device() -> <GpuBackend as BackendDevice>::Device {
    <GpuBackend as BackendDevice>::device()
}

// ── GPU availability cache ────────────────────────────────────────

static GPU_AVAILABLE: OnceLock<bool> = OnceLock::new();

fn gpu_available_cached() -> bool {
    *GPU_AVAILABLE.get_or_init(|| probe_gpu_available())
}

fn probe_gpu_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        use burn::backend::cuda::CudaDevice;
        // Attempt to create a tiny tensor on the CUDA device.
        // If this succeeds, CUDA is available.
        let dev = CudaDevice::default();
        let t: Tensor<Autodiff<burn::backend::Cuda>, 1> = Tensor::from_floats([0.0f32], &dev);
        // Force evaluation by reading the data
        let _ = t.to_data();
        return true;
    }
    #[cfg(feature = "metal")]
    {
        use burn::backend::metal::MetalDevice;
        let dev = MetalDevice::default();
        let t: Tensor<Autodiff<burn::backend::Metal>, 1> = Tensor::from_floats([0.0f32], &dev);
        let _ = t.to_data();
        return true;
    }
    #[cfg(feature = "vulkan")]
    {
        use burn::backend::vulkan::VulkanDevice;
        let dev = VulkanDevice::default();
        let t: Tensor<Autodiff<burn::backend::Vulkan>, 1> = Tensor::from_floats([0.0f32], &dev);
        let _ = t.to_data();
        return true;
    }
    #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
    {
        false
    }
}

// ── Tensor enum ───────────────────────────────────────────────────

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

// ── Helpers ───────────────────────────────────────────────────────

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
// NIF Functions
// ═══════════════════════════════════════════════════════════════════

// ── Tensor Creation ───────────────────────────────────────────────

#[inline(never)]
#[rustler::nif]
fn nif_new_tensor(data: Vec<u8>, shape: Vec<usize>, dtype: String) -> ResourceArc<TensorResource> {
    let t = make_tensor_from_bytes(data, shape.clone(), dtype.clone());
    build_resource(t, shape, dtype)
}

#[inline(never)]
#[rustler::nif]
fn nif_pow_tensor(a: ResourceArc<TensorResource>, _exp: f32) -> ResourceArc<TensorResource> {
    build_resource(a.tensor.clone(), a.shape.clone(), a.dtype.clone())
}

#[inline(never)]
#[rustler::nif]
fn nif_empty_tensor(shape: Vec<usize>, dtype: String) -> ResourceArc<TensorResource> {
    let numel: usize = shape.iter().product();
    let data = vec![0u8; numel * 4];
    let t = make_tensor_from_bytes(data, shape.clone(), dtype.clone());
    build_resource(t, shape, dtype)
}

#[inline(never)]
#[rustler::nif]
fn nif_zeros_tensor(shape: Vec<usize>, dtype: String) -> ResourceArc<TensorResource> {
    let numel: usize = shape.iter().product();
    let data = vec![0u8; numel * 4];
    let t = make_tensor_from_bytes(data, shape.clone(), dtype.clone());
    build_resource(t, shape, dtype)
}

#[inline(never)]
#[rustler::nif]
fn nif_ones_tensor(shape: Vec<usize>, dtype: String) -> ResourceArc<TensorResource> {
    let numel: usize = shape.iter().product();
    let vals = vec![1.0f32; numel];
    let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let t = make_tensor_from_bytes(data, shape.clone(), dtype.clone());
    build_resource(t, shape, dtype)
}

#[inline(never)]
#[rustler::nif]
fn nif_eye_tensor(size: usize, _type: String) -> ResourceArc<TensorResource> {
    let dev = device();
    let t = Tensor::<B, 2>::eye(size, &dev);
    build_resource(BurnTensor::F32x2(t), vec![size, size], "f32".into())
}

#[inline(never)]
#[rustler::nif]
fn nif_iota_tensor(shape: Vec<usize>, axis: usize, _type: String) -> ResourceArc<TensorResource> {
    let dev = device();
    let n = shape.get(axis).copied().unwrap_or(1);
    let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let t = Tensor::<B, 1>::from_floats(vals.as_slice(), &dev);
    build_resource(BurnTensor::F32x1(t), vec![n], "f32".into())
}

// ── Tensor Inspection ─────────────────────────────────────────────

#[inline(never)]
#[rustler::nif]
fn nif_tensor_shape(tensor: ResourceArc<TensorResource>) -> Vec<usize> {
    tensor.shape.clone()
}

#[inline(never)]
#[rustler::nif]
fn nif_tensor_dtype(tensor: ResourceArc<TensorResource>) -> String {
    tensor.dtype.clone()
}

#[inline(never)]
#[rustler::nif]
fn nif_tensor_to_binary(tensor: ResourceArc<TensorResource>) -> Vec<u8> {
    let (_, _, bytes) = tensor_to_bytes(&tensor.tensor);
    bytes
}

#[inline(never)]
#[rustler::nif]
fn nif_tensor_numel(tensor: ResourceArc<TensorResource>) -> usize {
    tensor.shape.iter().product()
}

// ── Element-wise Arithmetic ───────────────────────────────────────

#[inline(never)]
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

#[inline(never)]
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

#[inline(never)]
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

#[inline(never)]
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

#[inline(never)]
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

#[inline(never)]
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

#[inline(never)]
#[rustler::nif]
fn nif_exp_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().exp()),
        _ => panic!("Unsupported tensor type for exp"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[inline(never)]
#[rustler::nif]
fn nif_log_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().log()),
        _ => panic!("Unsupported tensor type for log"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[inline(never)]
#[rustler::nif]
fn nif_sqrt_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().sqrt()),
        _ => panic!("Unsupported tensor type for sqrt"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[inline(never)]
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

#[inline(never)]
#[rustler::nif]
fn nif_tanh_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().tanh()),
        _ => panic!("Unsupported tensor type for tanh"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[inline(never)]
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

// ── Reductions ────────────────────────────────────────────────────

#[inline(never)]
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

#[inline(never)]
#[rustler::nif]
fn nif_mean_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().mean()),
        _ => panic!("Unsupported tensor type for mean"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[inline(never)]
#[rustler::nif]
fn nif_max_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().max()),
        _ => panic!("Unsupported tensor type for max"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[inline(never)]
#[rustler::nif]
fn nif_min_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().min()),
        _ => panic!("Unsupported tensor type for min"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

// ── Linear Algebra ────────────────────────────────────────────────

#[inline(never)]
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

#[inline(never)]
#[rustler::nif]
fn nif_transpose_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().transpose()),
        _ => panic!("Transpose requires 2D tensor"),
    };
    let (shape, dtype, _) = tensor_to_bytes(&result);
    build_resource(result, shape, dtype)
}

#[inline(never)]
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

// ── Shape Manipulation ────────────────────────────────────────────

#[inline(never)]
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

#[inline(never)]
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

#[inline(never)]
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

// ── Device Management ─────────────────────────────────────────────

#[inline(never)]
#[rustler::nif]
fn nif_gpu_available() -> bool {
    gpu_available_cached()
}

#[inline(never)]
#[rustler::nif]
fn nif_device_name() -> String {
    if gpu_available_cached() {
        #[cfg(feature = "cuda")]
        return "CUDA (NVIDIA GPU)".into();
        #[cfg(feature = "metal")]
        return "Metal (Apple GPU)".into();
        #[cfg(feature = "vulkan")]
        return "Vulkan (GPU)".into();
        #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
        return "NdArray (CPU)".into();
    } else {
        "NdArray (CPU)".into()
    }
}

#[inline(never)]
#[rustler::nif]
fn nif_to_gpu(tensor: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    // With a GPU backend compiled in, tensors are already on the GPU device.
    // This function is a no-op when the backend is GPU, but still forces
    // evaluation (synchronization) so the data is materialized.
    let (_, _, bytes) = tensor_to_bytes(&tensor.tensor);
    let t = make_tensor_from_bytes(bytes, tensor.shape.clone(), tensor.dtype.clone());
    build_resource(t, tensor.shape.clone(), tensor.dtype.clone())
}

#[inline(never)]
#[rustler::nif]
fn nif_to_cpu(tensor: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    // Materialize the tensor data (forces GPU→CPU transfer if on GPU),
    // then rebuild on the CPU device.
    let (_, _, bytes) = tensor_to_bytes(&tensor.tensor);

    let t = {
        // Read data back to host, then rebuild using the B backend device
        // (which may be GPU or CPU depending on compilation).
        make_tensor_from_bytes(bytes, tensor.shape.clone(), tensor.dtype.clone())
    };

    build_resource(t, tensor.shape.clone(), tensor.dtype.clone())
}

// ── Memory Management ─────────────────────────────────────────────

#[inline(never)]
#[rustler::nif]
fn nif_free_tensor(_tensor: ResourceArc<TensorResource>) -> rustler::Atom {
    rustler::types::atom::ok()
}

// ── Neural Network Operations ─────────────────────────────────────

#[inline(never)]
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

#[inline(never)]
#[rustler::nif]
fn nif_layer_norm_tensor(
    a: ResourceArc<TensorResource>,
    _dim: i64,
    _eps: f64,
) -> ResourceArc<TensorResource> {
    build_resource(a.tensor.clone(), a.shape.clone(), a.dtype.clone())
}

// ═══════════════════════════════════════════════════════════════════
// NIF init
// ═══════════════════════════════════════════════════════════════════

rustler::init!("Elixir.ExBurn.Nif");

#[no_mangle]
pub extern "C" fn debug_nif_count() -> i32 {
    rustler::codegen_runtime::inventory::iter::<rustler::Nif>()
        .into_iter()
        .count() as i32
}
