/// zCUDA: cuRAND API - Raw FFI bindings.
///
/// Layer 1: Direct @cImport of curand.h for GPU random number generation.
const std = @import("std");

pub const c = @cImport({
    @cInclude("curand.h");
});

// Core types
pub const curandStatus_t = c.curandStatus_t;
pub const curandGenerator_t = c.curandGenerator_t;
pub const curandRngType_t = c.curandRngType_t;

// RNG types
pub const CURAND_RNG_PSEUDO_DEFAULT = c.CURAND_RNG_PSEUDO_DEFAULT;
pub const CURAND_RNG_PSEUDO_XORWOW = c.CURAND_RNG_PSEUDO_XORWOW;
pub const CURAND_RNG_PSEUDO_MRG32K3A = c.CURAND_RNG_PSEUDO_MRG32K3A;
pub const CURAND_RNG_PSEUDO_MTGP32 = c.CURAND_RNG_PSEUDO_MTGP32;
pub const CURAND_RNG_PSEUDO_MT19937 = c.CURAND_RNG_PSEUDO_MT19937;
pub const CURAND_RNG_PSEUDO_PHILOX4_32_10 = c.CURAND_RNG_PSEUDO_PHILOX4_32_10;
pub const CURAND_RNG_QUASI_DEFAULT = c.CURAND_RNG_QUASI_DEFAULT;
pub const CURAND_RNG_QUASI_SOBOL32 = c.CURAND_RNG_QUASI_SOBOL32;
pub const CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = c.CURAND_RNG_QUASI_SCRAMBLED_SOBOL32;

// Status codes
pub const CURAND_STATUS_SUCCESS = c.CURAND_STATUS_SUCCESS;
pub const CURAND_STATUS_VERSION_MISMATCH = c.CURAND_STATUS_VERSION_MISMATCH;
pub const CURAND_STATUS_NOT_INITIALIZED = c.CURAND_STATUS_NOT_INITIALIZED;
pub const CURAND_STATUS_ALLOCATION_FAILED = c.CURAND_STATUS_ALLOCATION_FAILED;
pub const CURAND_STATUS_TYPE_ERROR = c.CURAND_STATUS_TYPE_ERROR;
pub const CURAND_STATUS_OUT_OF_RANGE = c.CURAND_STATUS_OUT_OF_RANGE;
pub const CURAND_STATUS_LENGTH_NOT_MULTIPLE = c.CURAND_STATUS_LENGTH_NOT_MULTIPLE;
pub const CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = c.CURAND_STATUS_DOUBLE_PRECISION_REQUIRED;
pub const CURAND_STATUS_LAUNCH_FAILURE = c.CURAND_STATUS_LAUNCH_FAILURE;
pub const CURAND_STATUS_PREEXISTING_FAILURE = c.CURAND_STATUS_PREEXISTING_FAILURE;
pub const CURAND_STATUS_INITIALIZATION_FAILED = c.CURAND_STATUS_INITIALIZATION_FAILED;
pub const CURAND_STATUS_ARCH_MISMATCH = c.CURAND_STATUS_ARCH_MISMATCH;
pub const CURAND_STATUS_INTERNAL_ERROR = c.CURAND_STATUS_INTERNAL_ERROR;

// Core functions
pub const curandCreateGenerator = c.curandCreateGenerator;
pub const curandDestroyGenerator = c.curandDestroyGenerator;
pub const curandSetPseudoRandomGeneratorSeed = c.curandSetPseudoRandomGeneratorSeed;
pub const curandSetStream = c.curandSetStream;
pub const curandSetGeneratorOffset = c.curandSetGeneratorOffset;
pub const curandGenerateUniform = c.curandGenerateUniform;
pub const curandGenerateUniformDouble = c.curandGenerateUniformDouble;
pub const curandGenerateNormal = c.curandGenerateNormal;
pub const curandGenerateNormalDouble = c.curandGenerateNormalDouble;
pub const curandGenerate = c.curandGenerate;
pub const curandGenerateLogNormal = c.curandGenerateLogNormal;
pub const curandGenerateLogNormalDouble = c.curandGenerateLogNormalDouble;
pub const curandSetQuasiRandomGeneratorDimensions = c.curandSetQuasiRandomGeneratorDimensions;
pub const curandGeneratePoisson = c.curandGeneratePoisson;
pub const curandCreateGeneratorHost = c.curandCreateGeneratorHost;
