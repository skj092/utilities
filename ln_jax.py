import jax
import jax.numpy as jnp


def layer_norm_forward(x, w, b, eps=1e-5):
    """Forward pass for Layer Normalization"""
    mean = jnp.mean(
        x, axis=-1, keepdims=True)  # Compute mean along last dimension
    # Compute variance along last dimension
    var = jnp.var(x, axis=-1, keepdims=True)
    rstd = 1.0 / jnp.sqrt(var + eps)            # Reciprocal standard deviation
    norm = (x - mean) * rstd                    # Normalize x
    out = norm * w + b                          # Apply affine transform
    cache = (x, w, mean, rstd, norm)            # Store for backward pass
    return out, cache


def layer_norm_backward(dout, cache):
    """Backward pass for Layer Normalization"""
    x, w, mean, rstd, norm = cache
    B, T, C = x.shape  # Batch, Time, Channels

    db = jnp.sum(dout, axis=(0, 1))  # Sum gradients for bias
    dw = jnp.sum(dout * norm, axis=(0, 1))  # Sum gradients for weights

    dnorm = dout * w  # Gradient w.r.t. norm
    dx = dnorm - jnp.mean(dnorm, axis=-1, keepdims=True) - \
        norm * jnp.mean(dnorm * norm, axis=-1, keepdims=True)
    dx *= rstd  # Scale by reciprocal std deviation

    return dx, dw, db


# Sample Input (2x3x3)
x = jnp.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
               [[2, 4, 6], [8, 10, 12], [14, 16, 18]]], dtype=jnp.float32)

# Weights and Bias (Initialized to 1 and 0 for simplicity)
w = jnp.ones((1, 3))  # Same shape as last dimension
b = jnp.zeros((1, 3))

# Forward Pass
out, cache = layer_norm_forward(x, w, b)
print("Forward Output:\n", out)

# Backward Pass (Using example dout)
dout = jnp.ones_like(x)  # Assume uniform gradient for testing
dx, dw, db = layer_norm_backward(dout, cache)

print("\nGradients:")
print("dx:\n", dx)
print("dw:\n", dw)
print("db:\n", db)

# Verify gradients using JAX automatic differentiation


def loss_fn(x, w, b):
    out, _ = layer_norm_forward(x, w, b)
    return jnp.sum(out)  # Example loss function


grads = jax.grad(loss_fn, argnums=(0, 1, 2))(x, w, b)
print("\nJAX Autograd Check:")
print("dx (JAX):\n", grads[0])
print("dw (JAX):\n", grads[1])
print("db (JAX):\n", grads[2])
