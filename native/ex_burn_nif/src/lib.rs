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

use rustler::Binary;
use rustler::ResourceArc;
use std::cell::RefCell;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::OnceLock;

use burn::prelude::ElementConversion;
use burn::prelude::TensorData;
use burn::tensor::backend::AutodiffBackend;
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
    *GPU_AVAILABLE.get_or_init(probe_gpu_available)
}

fn probe_gpu_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        // cudarc panics (rather than erroring) when libcuda is missing,
        // which would abort the NIF. Catch the panic and report no GPU.
        let result = std::panic::catch_unwind(|| {
            use burn::backend::cuda::CudaDevice;
            // Attempt to create a tiny tensor on the CUDA device.
            // If this succeeds, CUDA is available.
            let dev = CudaDevice::default();
            let t: Tensor<Autodiff<burn::backend::Cuda>, 1> = Tensor::from_floats([0.0f32], &dev);
            // Force evaluation by reading the data
            let _ = t.to_data();
            true
        });
        result.unwrap_or(false)
    }
    #[cfg(feature = "metal")]
    {
        // Metal initialization can panic on devices without a GPU or when
        // the framework is unavailable — catch and report no GPU.
        let result = std::panic::catch_unwind(|| {
            use burn::backend::metal::MetalDevice;
            let dev = MetalDevice::default();
            let t: Tensor<Autodiff<burn::backend::Metal>, 1> = Tensor::from_floats([0.0f32], &dev);
            let _ = t.to_data();
            true
        });
        result.unwrap_or(false)
    }
    #[cfg(feature = "vulkan")]
    {
        // Vulkan initialization can panic when no compatible device or
        // loader is present — catch and report no GPU.
        let result = std::panic::catch_unwind(|| {
            use burn::backend::vulkan::VulkanDevice;
            let dev = VulkanDevice::default();
            let t: Tensor<Autodiff<burn::backend::Vulkan>, 1> = Tensor::from_floats([0.0f32], &dev);
            let _ = t.to_data();
            true
        });
        result.unwrap_or(false)
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
    let rank = if shape.is_empty() { 1 } else { shape.len() };
    let padded: [usize; 4] = [
        *shape.first().unwrap_or(&1),
        *shape.get(1).unwrap_or(&1),
        *shape.get(2).unwrap_or(&1),
        *shape.get(3).unwrap_or(&1),
    ];
    let dims: Vec<usize> = padded[..rank].to_vec();
    let data = TensorData::new(vals.to_vec(), dims);
    match rank {
        // Freshly created tensors are autodiff leaves: track them so that
        // subsequent operations build a graph usable by `backward()`.
        1 => BurnTensor::F32x1(Tensor::<B, 1>::from_data(data, &dev).require_grad()),
        2 => BurnTensor::F32x2(Tensor::<B, 2>::from_data(data, &dev).require_grad()),
        3 => BurnTensor::F32x3(Tensor::<B, 3>::from_data(data, &dev).require_grad()),
        4 => BurnTensor::F32x4(Tensor::<B, 4>::from_data(data, &dev).require_grad()),
        _ => panic!("Unsupported rank {}", rank),
    }
}

fn make_tensor_from_bytes(data: Vec<u8>, shape: Vec<usize>, dtype: String) -> BurnTensor {
    match dtype.as_str() {
        "f32" => {
            let vals: Vec<f32> = data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
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

/// Wraps an operation result into a resource without reading tensor data.
///
/// Shape is taken from the tensor's metadata, so no GPU→CPU sync happens
/// here. Data is only materialized by `nif_tensor_to_binary`.
fn wrap_tensor(t: BurnTensor) -> ResourceArc<TensorResource> {
    let shape = match &t {
        BurnTensor::F32x1(t) => t.shape().to_vec(),
        BurnTensor::F32x2(t) => t.shape().to_vec(),
        BurnTensor::F32x3(t) => t.shape().to_vec(),
        BurnTensor::F32x4(t) => t.shape().to_vec(),
    };
    build_resource(t, shape, "f32".into())
}

// ═══════════════════════════════════════════════════════════════════
// NIF Functions
// ═══════════════════════════════════════════════════════════════════

// ── Tensor Creation ───────────────────────────────────────────────

#[rustler::nif]
fn nif_new_tensor(data: Binary, shape: Vec<usize>, dtype: String) -> ResourceArc<TensorResource> {
    let t = make_tensor_from_bytes(data.as_slice().to_vec(), shape.clone(), dtype.clone());
    build_resource(t, shape, dtype)
}

#[rustler::nif]
fn nif_pow_tensor(a: ResourceArc<TensorResource>, exp: f32) -> ResourceArc<TensorResource> {
    // Element-wise power with a scalar exponent
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().powf_scalar(exp)),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().powf_scalar(exp)),
        _ => panic!("pow_tensor only supports f32 1D/2D tensors"),
    };
    wrap_tensor(result)
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

// ── Tensor Inspection ─────────────────────────────────────────────

#[rustler::nif]
fn nif_tensor_shape(tensor: ResourceArc<TensorResource>) -> Vec<usize> {
    tensor.shape.clone()
}

#[rustler::nif]
fn nif_tensor_dtype(tensor: ResourceArc<TensorResource>) -> String {
    tensor.dtype.clone()
}

#[rustler::nif]
fn nif_tensor_to_binary<'a>(
    env: rustler::Env<'a>,
    tensor: ResourceArc<TensorResource>,
) -> Binary<'a> {
    let (_, _, bytes) = tensor_to_bytes(&tensor.tensor);
    let mut owned = rustler::types::OwnedBinary::new(bytes.len()).expect("alloc binary");
    owned.as_mut_slice().copy_from_slice(&bytes);
    Binary::from_owned(owned, env)
}

#[rustler::nif]
fn nif_tensor_numel(tensor: ResourceArc<TensorResource>) -> usize {
    tensor.shape.iter().product()
}

// ── Element-wise Arithmetic ───────────────────────────────────────

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
    wrap_tensor(result)
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
    wrap_tensor(result)
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
    wrap_tensor(result)
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
    wrap_tensor(result)
}

#[rustler::nif]
fn nif_neg_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(-t.clone()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(-t.clone()),
        _ => panic!("Unsupported tensor type for neg"),
    };
    wrap_tensor(result)
}

#[rustler::nif]
fn nif_abs_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().abs()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().abs()),
        _ => panic!("Unsupported tensor type for abs"),
    };
    wrap_tensor(result)
}

#[rustler::nif]
fn nif_exp_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().exp()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().exp()),
        _ => panic!("Unsupported tensor type for exp"),
    };
    wrap_tensor(result)
}

#[rustler::nif]
fn nif_log_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().log()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().log()),
        _ => panic!("Unsupported tensor type for log"),
    };
    wrap_tensor(result)
}

#[rustler::nif]
fn nif_sqrt_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().sqrt()),
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().sqrt()),
        _ => panic!("Unsupported tensor type for sqrt"),
    };
    wrap_tensor(result)
}

#[rustler::nif]
fn sigmoid_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let dev = device();
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => {
            let one = Tensor::<B, 1>::ones(t.shape(), &dev);
            BurnTensor::F32x1(one.clone() / (one + (-t.clone()).exp()))
        }
        BurnTensor::F32x2(t) => {
            let one = Tensor::<B, 2>::ones(t.shape(), &dev);
            BurnTensor::F32x2(one.clone() / (one + (-t.clone()).exp()))
        }
        _ => panic!("Unsupported tensor type for sigmoid"),
    };
    wrap_tensor(result)
}

#[rustler::nif]
fn nif_tanh_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().tanh()),
        _ => panic!("Unsupported tensor type for tanh"),
    };
    wrap_tensor(result)
}

#[rustler::nif]
fn nif_relu_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let dev = device();
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => {
            let z = Tensor::<B, 1>::zeros(t.shape(), &dev);
            BurnTensor::F32x1(t.clone().max_pair(z))
        }
        BurnTensor::F32x2(t) => {
            let z = Tensor::<B, 2>::zeros(t.shape(), &dev);
            BurnTensor::F32x2(t.clone().max_pair(z))
        }
        _ => panic!("Unsupported tensor type for relu"),
    };
    wrap_tensor(result)
}

// ── Reductions ────────────────────────────────────────────────────

#[rustler::nif]
fn nif_sum_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().sum()),
        BurnTensor::F32x2(t) => BurnTensor::F32x1(t.clone().sum()),
        _ => panic!("Unsupported tensor type for sum"),
    };
    wrap_tensor(result)
}

#[rustler::nif]
fn nif_mean_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().mean()),
        BurnTensor::F32x2(t) => BurnTensor::F32x1(t.clone().mean()),
        _ => panic!("Unsupported tensor type for mean"),
    };
    wrap_tensor(result)
}

#[rustler::nif]
fn nif_max_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().max()),
        _ => panic!("Unsupported tensor type for max"),
    };
    wrap_tensor(result)
}

#[rustler::nif]
fn nif_min_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x1(t) => BurnTensor::F32x1(t.clone().min()),
        _ => panic!("Unsupported tensor type for min"),
    };
    wrap_tensor(result)
}

// ── Linear Algebra ────────────────────────────────────────────────

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
    wrap_tensor(result)
}

#[rustler::nif]
fn nif_transpose_tensor(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    let result = match &a.tensor {
        BurnTensor::F32x2(t) => BurnTensor::F32x2(t.clone().transpose()),
        _ => panic!("Transpose requires 2D tensor"),
    };
    wrap_tensor(result)
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
    wrap_tensor(result)
}

// ── Shape Manipulation ────────────────────────────────────────────

#[rustler::nif]
fn nif_reshape_tensor(
    a: ResourceArc<TensorResource>,
    new_shape: Vec<usize>,
) -> ResourceArc<TensorResource> {
    let new_numel: usize = new_shape.iter().product::<usize>().max(1);
    let old_numel: usize = a.shape.iter().product();
    if new_numel != old_numel {
        panic!("Cannot reshape {} elements into {:?}", old_numel, new_shape);
    }
    // Rank-0 (scalar) results are represented as a 1-element rank-1 tensor.
    let effective_shape: Vec<usize> = if new_shape.is_empty() {
        vec![1]
    } else {
        new_shape.clone()
    };
    let result = match (&a.tensor, effective_shape.as_slice()) {
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
    wrap_tensor(result)
}

/// Broadcasts a tensor to `target_shape` following NumPy/Nx rules:
/// source dims are right-aligned, singleton dims are expanded.
fn broadcast_burn_tensor(tensor: &BurnTensor, target_shape: &[usize]) -> BurnTensor {
    let src_dims: Vec<usize> = match tensor {
        BurnTensor::F32x1(t) => t.shape().to_vec(),
        BurnTensor::F32x2(t) => t.shape().to_vec(),
        BurnTensor::F32x3(t) => t.shape().to_vec(),
        BurnTensor::F32x4(t) => t.shape().to_vec(),
    };
    if target_shape.len() < src_dims.len() {
        panic!(
            "broadcast: target rank {} is lower than source rank {}",
            target_shape.len(),
            src_dims.len()
        );
    }
    // Align trailing dimensions by prepending singleton axes, reshape to the
    // padded shape (element count unchanged), then expand.
    let mut padded = vec![1usize; target_shape.len() - src_dims.len()];
    padded.extend_from_slice(&src_dims);

    let reshaped: BurnTensor = match (tensor, padded.as_slice()) {
        (BurnTensor::F32x1(t), _) => BurnTensor::F32x1(t.clone()),
        (BurnTensor::F32x2(t), [d0, d1]) => BurnTensor::F32x2(t.clone().reshape([*d0, *d1])),
        (BurnTensor::F32x3(t), [d0, d1, d2]) => {
            BurnTensor::F32x3(t.clone().reshape([*d0, *d1, *d2]))
        }
        (BurnTensor::F32x4(t), [d0, d1, d2, d3]) => {
            BurnTensor::F32x4(t.clone().reshape([*d0, *d1, *d2, *d3]))
        }
        _ => panic!("broadcast: unsupported source/target rank combination"),
    };

    match (reshaped, target_shape) {
        (t @ BurnTensor::F32x1(_), [_d0]) => t,
        (BurnTensor::F32x1(t), [d0, d1]) => BurnTensor::F32x2(t.expand([*d0, *d1])),
        (BurnTensor::F32x2(t), [d0, d1]) => BurnTensor::F32x2(t.expand([*d0, *d1])),
        (BurnTensor::F32x2(t), [d0, d1, d2]) => BurnTensor::F32x3(t.expand([*d0, *d1, *d2])),
        (BurnTensor::F32x3(t), [d0, d1, d2]) => BurnTensor::F32x3(t.expand([*d0, *d1, *d2])),
        (BurnTensor::F32x3(t), [d0, d1, d2, d3]) => {
            BurnTensor::F32x4(t.expand([*d0, *d1, *d2, *d3]))
        }
        (BurnTensor::F32x4(t), [d0, d1, d2, d3]) => {
            BurnTensor::F32x4(t.expand([*d0, *d1, *d2, *d3]))
        }
        _ => panic!("broadcast: unsupported target rank"),
    }
}

#[rustler::nif]
fn nif_broadcast_tensor(
    a: ResourceArc<TensorResource>,
    target_shape: Vec<usize>,
) -> ResourceArc<TensorResource> {
    wrap_tensor(broadcast_burn_tensor(&a.tensor, &target_shape))
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
    wrap_tensor(result)
}

// ── Device Management ─────────────────────────────────────────────

#[rustler::nif]
fn nif_gpu_available() -> bool {
    gpu_available_cached()
}

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

#[rustler::nif]
fn nif_to_gpu(tensor: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
    // With a GPU backend compiled in, tensors are already on the GPU device.
    // This function is a no-op when the backend is GPU, but still forces
    // evaluation (synchronization) so the data is materialized.
    let (_, _, bytes) = tensor_to_bytes(&tensor.tensor);
    let t = make_tensor_from_bytes(bytes, tensor.shape.clone(), tensor.dtype.clone());
    build_resource(t, tensor.shape.clone(), tensor.dtype.clone())
}

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

#[rustler::nif]
fn nif_free_tensor(_tensor: ResourceArc<TensorResource>) -> rustler::Atom {
    rustler::types::atom::ok()
}

// ── Neural Network Operations ─────────────────────────────────────

#[rustler::nif]
fn nif_softmax_tensor(a: ResourceArc<TensorResource>, dim: i64) -> ResourceArc<TensorResource> {
    let dim = if dim < 0 {
        a.shape.len().saturating_sub(1)
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
    wrap_tensor(result)
}

#[rustler::nif]
fn nif_layer_norm_tensor(
    a: ResourceArc<TensorResource>,
    dim: i64,
    eps: f64,
) -> ResourceArc<TensorResource> {
    let eps = eps as f32;
    let result = match &a.tensor {
        BurnTensor::F32x2(t) => {
            let norm_dim = if dim < 0 {
                a.shape.len().saturating_sub(1)
            } else {
                dim as usize
            };
            let mean = t.clone().mean_dim(norm_dim);
            let shifted = t.clone() - mean;
            let var = shifted.clone().powf_scalar(2.0_f32).mean_dim(norm_dim);
            let normalized = shifted / (var + eps).sqrt();
            BurnTensor::F32x2(normalized)
        }
        BurnTensor::F32x1(t) => {
            let mean = t.clone().mean();
            let shifted = t.clone() - mean;
            let var = shifted.clone().powf_scalar(2.0_f32).mean();
            let normalized = shifted / (var + eps).sqrt();
            BurnTensor::F32x1(normalized)
        }
        _ => panic!("layer_norm only supports 1D/2D f32 tensors"),
    };
    wrap_tensor(result)
}

// ═══════════════════════════════════════════════════════════════════
// NIF init
// ═══════════════════════════════════════════════════════════════════

type GradientsType = <B as AutodiffBackend>::Gradients;

thread_local! {
    static LAST_GRADIENTS: RefCell<Option<GradientsType>> = const { RefCell::new(None) };
}

// ── Autodiff / Backward ────────────────────────────────────────-

/// Triggers backward pass on a scalar tensor and stores gradients
/// in a thread-local for subsequent grad() calls.
#[rustler::nif]
fn nif_backward_tensor(a: ResourceArc<TensorResource>) -> rustler::Atom {
    if let BurnTensor::F32x1(t) = &a.tensor {
        if a.shape.iter().product::<usize>() == 1 {
            let grads = t.clone().backward();
            LAST_GRADIENTS.with(|g| {
                *g.borrow_mut() = Some(grads);
            });
        }
    }

    rustler::types::atom::ok()
}

#[rustler::nif]
fn nif_grad_tensor(
    tensor: ResourceArc<TensorResource>,
    _var: ResourceArc<TensorResource>,
) -> ResourceArc<TensorResource> {
    let grad_tensor = LAST_GRADIENTS.with(|grads_cell| {
        let grads_ref = grads_cell.borrow();
        let grads = grads_ref.as_ref();

        match &tensor.tensor {
            BurnTensor::F32x1(t) => match grads {
                Some(g) => match t.grad(g) {
                    Some(grad) => {
                        let data = grad.to_data();
                        let vals: Vec<f32> = data.into_vec().unwrap_or_default();
                        let dev = device();
                        BurnTensor::F32x1(Tensor::<B, 1>::from_floats(vals.as_slice(), &dev))
                    }
                    None => {
                        let dev = device();
                        let n: usize = tensor.shape.iter().product();
                        BurnTensor::F32x1(Tensor::<B, 1>::zeros([n], &dev))
                    }
                },
                None => {
                    let dev = device();
                    let n: usize = tensor.shape.iter().product();
                    BurnTensor::F32x1(Tensor::<B, 1>::zeros([n], &dev))
                }
            },
            BurnTensor::F32x2(t) => match grads {
                Some(g) => match t.grad(g) {
                    Some(grad) => {
                        let data = grad.to_data();
                        let vals: Vec<f32> = data.into_vec().unwrap_or_default();
                        let dev = device();
                        let dims = tensor.shape.clone();
                        BurnTensor::F32x2(Tensor::<B, 2>::from_data(
                            TensorData::new(vals, dims),
                            &dev,
                        ))
                    }
                    None => {
                        let dev = device();
                        let [rows, cols]: [usize; 2] = tensor.shape.to_vec().try_into().unwrap();
                        BurnTensor::F32x2(Tensor::<B, 2>::zeros([rows, cols], &dev))
                    }
                },
                None => {
                    let dev = device();
                    let [rows, cols]: [usize; 2] = tensor.shape.to_vec().try_into().unwrap();
                    BurnTensor::F32x2(Tensor::<B, 2>::zeros([rows, cols], &dev))
                }
            },
            _ => panic!("grad_tensor only supports 1D/2D f32 tensors"),
        }
    });
    wrap_tensor(grad_tensor)
}

// ═══════════════════════════════════════════════════════════════════
// NIF init
// ═══════════════════════════════════════════════════════════════════

rustler::init!("Elixir.ExBurn.Nif");

// ── Loss Functions ──────────────────────────────────────────────

#[rustler::nif]
fn nif_cross_entropy_loss(
    pred: ResourceArc<TensorResource>,
    target: ResourceArc<TensorResource>,
) -> ResourceArc<TensorResource> {
    let result = match (&pred.tensor, &target.tensor) {
        (BurnTensor::F32x2(logits), BurnTensor::F32x2(targets)) => {
            let max_val = logits.clone().max_dim(1);
            let shifted = logits.clone() - max_val;
            let exp_vals = shifted.clone().exp();
            let sum_exp = exp_vals.clone().sum_dim(1);
            let log_sum_exp = sum_exp.log();
            let log_probs = shifted - log_sum_exp;
            let batch_size = log_probs.shape().dims::<2>()[0];
            let nll = -(targets.clone() * log_probs).sum() / (batch_size as f32);
            let dev = device();
            BurnTensor::F32x1(Tensor::<B, 1>::from_floats(
                [nll.into_scalar().elem::<f32>()],
                &dev,
            ))
        }
        (BurnTensor::F32x1(logits), BurnTensor::F32x1(targets)) => {
            let max_val = logits.clone().max();
            let shifted = logits.clone() - max_val;
            let exp_vals = shifted.clone().exp();
            let sum_exp = exp_vals.clone().sum();
            let log_sum_exp = sum_exp.log();
            let log_probs = shifted - log_sum_exp;
            let nll = -(targets.clone() * log_probs).sum();
            let dev = device();
            BurnTensor::F32x1(Tensor::<B, 1>::from_floats(
                [nll.into_scalar().elem::<f32>()],
                &dev,
            ))
        }
        _ => panic!("cross_entropy_loss: unsupported tensor shapes"),
    };
    wrap_tensor(result)
}

#[rustler::nif]
fn nif_mse_loss(
    pred: ResourceArc<TensorResource>,
    target: ResourceArc<TensorResource>,
) -> ResourceArc<TensorResource> {
    let result = match (&pred.tensor, &target.tensor) {
        (BurnTensor::F32x1(p), BurnTensor::F32x1(t)) => {
            let diff = p.clone() - t.clone();
            let squared = diff.clone() * diff;
            let mse = squared.mean();
            let dev = device();
            BurnTensor::F32x1(Tensor::<B, 1>::from_floats(
                [mse.into_scalar().elem::<f32>()],
                &dev,
            ))
        }
        (BurnTensor::F32x2(p), BurnTensor::F32x2(t)) => {
            let diff = p.clone() - t.clone();
            let squared = diff.clone() * diff;
            let mse = squared.mean();
            let dev = device();
            BurnTensor::F32x1(Tensor::<B, 1>::from_floats(
                [mse.into_scalar().elem::<f32>()],
                &dev,
            ))
        }
        _ => panic!("mse_loss: unsupported tensor shapes"),
    };
    wrap_tensor(result)
}

// ── Regularization ──────────────────────────────────────────────

#[rustler::nif]
fn nif_dropout(tensor: ResourceArc<TensorResource>, prob: f64) -> ResourceArc<TensorResource> {
    let prob = prob as f32;
    let scale = 1.0 / (1.0 - prob);
    let dev = device();

    let result = match &tensor.tensor {
        BurnTensor::F32x2(t) => {
            let dims = t.shape().dims::<2>();
            let total = dims[0] * dims[1];
            let vals: Vec<f32> = (0..total)
                .map(|_| {
                    let r: f32 = rand::random();
                    if r > prob {
                        scale
                    } else {
                        0.0
                    }
                })
                .collect();
            let mask = Tensor::<B, 2>::from_data(TensorData::new(vals, dims.to_vec()), &device());
            BurnTensor::F32x2(t.clone() * mask)
        }
        BurnTensor::F32x1(t) => {
            let dims = t.shape().dims::<1>();
            let total = dims[0];
            let vals: Vec<f32> = (0..total)
                .map(|_| {
                    let r: f32 = rand::random();
                    if r > prob {
                        scale
                    } else {
                        0.0
                    }
                })
                .collect();
            let mask = Tensor::<B, 1>::from_floats(vals.as_slice(), &dev);
            BurnTensor::F32x1(t.clone() * mask)
        }
        _ => panic!("dropout: unsupported tensor type"),
    };
    wrap_tensor(result)
}

#[rustler::nif]
fn debug_nif_count() -> i32 {
    rustler::codegen_runtime::inventory::iter::<rustler::Nif>().count() as i32
}
