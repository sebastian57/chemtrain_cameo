# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A collection of functions to facilitate learning maximum likelihood /
 single point estimate models.
 """
from functools import partial
import os
import time

import jax
from jax import (lax, vmap, pmap, value_and_grad, tree_map, device_count,
                 numpy as jnp, device_put, jit)
from jax.sharding import Mesh, PartitionSpec, NamedSharding, SingleDeviceSharding
from jax.experimental import multihost_utils
from jax.experimental.shard_map import shard_map
from jax_sgmc import data
import optax

from chemtrain import util


def _str_to_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _dtype_from_name(name):
    name = str(name).strip().lower()
    if name == "bfloat16":
        return jnp.bfloat16
    if name in ("float32", "fp32"):
        return jnp.float32
    raise ValueError(
        f"Unsupported dtype name '{name}'. Expected one of: float32, bfloat16."
    )


def _cast_tree_floating(tree, dtype):
    """Cast floating leaves in a pytree to dtype."""
    def _cast_leaf(x):
        x_dtype = getattr(x, "dtype", None)
        if x_dtype is not None and jnp.issubdtype(x_dtype, jnp.inexact):
            return jnp.asarray(x, dtype=dtype)
        return x
    return tree_map(_cast_leaf, tree)


def _cast_grad_like_params(grad, params):
    """Cast grad leaves to parameter dtypes where needed."""
    def _cast_leaf(g, p):
        g_dtype = getattr(g, "dtype", None)
        p_dtype = getattr(p, "dtype", None)
        if (
            g_dtype is not None
            and p_dtype is not None
            and jnp.issubdtype(g_dtype, jnp.inexact)
            and jnp.issubdtype(p_dtype, jnp.inexact)
            and g_dtype != p_dtype
        ):
            return jnp.asarray(g, dtype=p_dtype)
        return g
    return tree_map(_cast_leaf, grad, params)


def _shape_vec(x, max_rank=6):
    """Return a fixed-length int32 vector with tensor shape entries."""
    shape = list(getattr(x, "shape", ()))
    if len(shape) < max_rank:
        shape = shape + [-1] * (max_rank - len(shape))
    else:
        shape = shape[:max_rank]
    return jnp.asarray(shape, dtype=jnp.int32)


def _put_process_local_data(data, mesh, pspec):
    """Convert per-process local data into a global sharded array when needed."""
    if mesh.size <= 1:
        return data

    sharding = NamedSharding(mesh, pspec)
    if jax.process_count() > 1:
        return multihost_utils.host_local_array_to_global_array(data, mesh, pspec)
    return device_put(data, sharding)


def _get_param_loss_fn(loss_fn, batched_model, penalty_fn=None):

    def params_loss_fn(params, batch, sample_mask=None):
        predictions = batched_model(params, batch)

        if sample_mask is None:
            out = loss_fn(predictions, batch)
        else:
            # Compute the loss for each sample to enable masking
            out = vmap(loss_fn)(predictions, batch)
            out = tree_map(partial(_batch_masked_loss, mask=sample_mask), out)

        # Canonicalize output
        if isinstance(out, tuple):
            loss, per_target_loss = out
        else:
            loss = out
            per_target_loss = None

        # Add a penalty if provided
        if penalty_fn is not None:
            loss += penalty_fn(params)

        return loss, per_target_loss
    return params_loss_fn


def pmap_update_fn(batched_model, loss_fn, optimizer, penalty_fn=None):
    """Initializes a pmapped function for updating parameters.

    Usage:
        .. code-block :: python

            params, opt_state, loss, grad = update_fn(params, opt_state, batch)

    Loss and grad are only a single instance, no n_device replica.
    Params and opt_state need to be N_devices times duplicated along axis 0.
    Batch is reshaped by this function.

    Args:
        batched_model: A model with signature model(params, batch), which
            predicts a batch of outputs used in loss function.
        loss_fn: Loss function(predictions, targets) returning the scalar loss
            value for a batch.
        optimizer: Optax optimizer
        penalty_fn: A penalty function based on the model parameters.

    Returns:
        A function that computes the gradient and updates the parameters via the
        optimizer.
    """
    # loss as function of params and batch for optimization
    param_loss_fn = _get_param_loss_fn(loss_fn, batched_model, penalty_fn)
    reduce_dtype = _dtype_from_name(os.environ.get("CHEMTRAIN_REDUCE_DTYPE", "float32"))

    @partial(jax.pmap, in_axes=(None, None, 0), axis_name='batch')
    def pmap_batch_update(params, opt_state, batch):
        (loss, per_target_loss), grad = value_and_grad(
            param_loss_fn, has_aux=True
        )(params, batch)

        # step optimizer within pmap to minimize communication overhead
        grad = _cast_tree_floating(grad, reduce_dtype)
        loss = jnp.asarray(loss, dtype=reduce_dtype)
        per_target_loss = _cast_tree_floating(per_target_loss, reduce_dtype)
        grad = lax.pmean(grad, 'batch')
        loss = lax.pmean(loss, 'batch')
        per_target_loss = lax.psum(per_target_loss, 'batch')
        grad = _cast_grad_like_params(grad, params)

        new_params, opt_state = step_optimizer(params, opt_state, grad,
                                               optimizer)
        return new_params, opt_state, loss, grad, per_target_loss

    def batch_update(params, opt_state, batch, per_target=False,
                     microbatch_count=1, **kwargs):
        if "per_target_loss" in kwargs:
            per_target = kwargs["per_target_loss"]
        if microbatch_count != 1:
            raise NotImplementedError(
                "microbatch_count > 1 is only implemented for shmap_update_fn."
            )
        batch = util.tree_pmap_split(batch, jax.local_device_count())
        out = pmap_batch_update(params, opt_state, batch)
        new_params, opt_state, loss, grad, per_target_loss = util.tree_get_single(out)

        if per_target:
            return new_params, opt_state, loss, grad, per_target_loss
        else:
            return new_params, opt_state, loss, grad
    return batch_update


def pmap_loss_fn(batched_model, loss_fn, penalty_fn=None):
    """Initializes a pmapped function for computing a loss.

    Usage:
        .. code-block :: python

            loss, per_target_losses = loss_fn(params, batch, per_target=True)

    Args:
        batched_model: A model with signature model(params, batch), which
            predicts a batch of outputs used in loss function.
        loss_fn: Loss function(predictions, targets) returning the scalar loss
            value for a batch.
        penalty_fn: A penalty function based on the model parameters.

    Returns:
        A function that computes the total loss and per-target loss
        contributions.
    """
    param_loss_fn = _get_param_loss_fn(loss_fn, batched_model, penalty_fn)

    @partial(jax.pmap, in_axes=(None, 0), axis_name='batch')
    def pmap_batch_loss(params, data):
        loss, per_target_loss = param_loss_fn(params, *data)
        loss = lax.pmean(loss, 'batch')
        per_target_loss = lax.pmean(per_target_loss, 'batch')
        return loss, per_target_loss

    def batch_loss(params, batch, mask=None, per_target=False):
        data = batch, mask
        data = util.tree_pmap_split(data, jax.local_device_count())
        out = pmap_batch_loss(params, data)
        loss, per_target_loss = util.tree_get_single(out)

        if per_target:
            return loss, per_target_loss
        else:
            return loss

    return batch_loss


def shmap_update_fn(batched_model, loss_fn, optimizer, penalty_fn=None):
    """Initializes a shmapped function for updating parameters.

    Usage:
        .. code-block :: python

            params, opt_state, loss, grad = update_fn(params, opt_state, batch)

    Args:
        batched_model: A model with signature model(params, batch), which
            predicts a batch of outputs used in loss function.
        loss_fn: Loss function(predictions, targets) returning the scalar loss
            value for a batch.
        optimizer: Optax optimizer
        penalty_fn: A penalty function based on the model parameters.

    Returns:
        A function that computes the gradient and updates the parameters via the
        optimizer.
    """
    # loss as function of params and batch for optimization.
    mesh = Mesh(jax.devices(), axis_names=('batch'))
    replicate = NamedSharding(mesh, PartitionSpec())
    profile_internal = _str_to_bool(
        os.environ.get("CHEMTRAIN_PROFILE_UPDATE_FN_INTERNAL", "0")
    )
    profile_block = _str_to_bool(
        os.environ.get("CHEMTRAIN_PROFILE_UPDATE_FN_INTERNAL_BLOCK", "0")
    )
    profile_rank0_only = _str_to_bool(
        os.environ.get("CHEMTRAIN_PROFILE_UPDATE_FN_INTERNAL_RANK0_ONLY", "1")
    )
    profile_components = _str_to_bool(
        os.environ.get("CHEMTRAIN_PROFILE_UPDATE_FN_COMPONENTS", "0")
    )
    profile_local_split = _str_to_bool(
        os.environ.get("CHEMTRAIN_PROFILE_UPDATE_FN_LOCAL_SPLIT", "0")
    )
    reduce_dtype = _dtype_from_name(os.environ.get("CHEMTRAIN_REDUCE_DTYPE", "float32"))
    enable_buffer_donation = _str_to_bool(
        os.environ.get("CHEMTRAIN_ENABLE_BUFFER_DONATION", "0")
    )
    donate_mode = str(os.environ.get("CHEMTRAIN_DONATE_MODE", "state_only")).strip().lower()
    if donate_mode not in ("state_only", "state_and_batch"):
        raise ValueError(
            f"Unsupported CHEMTRAIN_DONATE_MODE='{donate_mode}'. "
            "Expected one of: state_only, state_and_batch."
        )
    donate_state_argnums = (0, 1)
    donate_state_batch_argnums = (0, 1, 2)
    profile_limit = int(
        os.environ.get("CHEMTRAIN_PROFILE_UPDATE_FN_INTERNAL_LIMIT", "0")
    )
    compile_sig_debug = _str_to_bool(
        os.environ.get("CHEMTRAIN_DEBUG_COMPILE_SIGNATURE", "1")
    )
    compile_sig_rank0_only = _str_to_bool(
        os.environ.get("CHEMTRAIN_DEBUG_COMPILE_SIGNATURE_RANK0_ONLY", "1")
    )
    seen_shape_signatures = set()
    profile_rank = jax.process_index()
    profile_step = [0]
    printed_meta = [False]

    param_loss_fn = _get_param_loss_fn(loss_fn, batched_model, penalty_fn)

    batch_update_fns = {}
    component_fns = {}
    grad_accum_mode_default = str(
        os.environ.get("CHEMTRAIN_GRAD_ACCUM_MODE", "stack_scan")
    ).strip().lower()
    if grad_accum_mode_default not in ("concat_slice", "stack_scan"):
        raise ValueError(
            "Unsupported CHEMTRAIN_GRAD_ACCUM_MODE="
            f"'{grad_accum_mode_default}'. Expected one of: "
            "concat_slice, stack_scan."
        )
    debug_microbatch_grad_norms = _str_to_bool(
        os.environ.get("CHEMTRAIN_DEBUG_MICROBATCH_GRAD_NORMS", "0")
    )
    debug_shape_trace = _str_to_bool(
        os.environ.get("CHEMTRAIN_DEBUG_SHAPE_TRACE", "0")
    )
    debug_shape_trace_printed = [False]

    def _resolve_accum_mode(accum_mode):
        if accum_mode is None:
            accum_mode = grad_accum_mode_default
        else:
            accum_mode = str(accum_mode).strip().lower()
        if accum_mode not in ("concat_slice", "stack_scan"):
            raise ValueError(
                f"Unsupported accum_mode='{accum_mode}'. "
                "Expected one of: concat_slice, stack_scan."
            )
        return accum_mode

    def _shape_signature(batch, accum_mode, microbatch_count):
        leaves = jax.tree_util.tree_leaves(batch)
        sig_leaves = []
        for leaf in leaves[:4]:
            sig_leaves.append((
                tuple(getattr(leaf, "shape", ())),
                str(getattr(leaf, "dtype", "unknown")),
            ))
        return (
            accum_mode,
            int(microbatch_count),
            tuple(sig_leaves),
            int(len(leaves)),
        )

    def _batch_in_spec(accum_mode, microbatch_count):
        if accum_mode == "stack_scan" and microbatch_count > 1:
            # Keep the microbatch axis replicated and shard the true batch axis.
            return PartitionSpec(None, 'batch')
        return PartitionSpec('batch')

    def _accumulated_local_grad(
        params, batch, microbatch_count, accum_mode, emit_shape_trace=False
    ):
        if microbatch_count < 1:
            raise ValueError(f"microbatch_count must be >= 1, got {microbatch_count}")
        if emit_shape_trace:
            leaf_shape = _shape_vec(jax.tree_util.tree_leaves(batch)[0])
            jax.debug.print(
                "[ShapeTrace][local] mode={} microbatch_count={} batch_leaf_shape={}",
                accum_mode,
                microbatch_count,
                leaf_shape,
            )
        if microbatch_count == 1:
            return value_and_grad(param_loss_fn, has_aux=True)(params, batch)

        if accum_mode == "concat_slice":
            local_batch_size = jax.tree_util.tree_leaves(batch)[0].shape[0]
            if local_batch_size % microbatch_count != 0:
                raise ValueError(
                    "Local batch size must be divisible by microbatch_count. "
                    f"Got local_batch_size={local_batch_size}, "
                    f"microbatch_count={microbatch_count}"
                )

            microbatch_size = local_batch_size // microbatch_count

            def _slice_microbatch(batch_pytree, i):
                start = i * microbatch_size
                return tree_map(
                    lambda arr: lax.dynamic_slice_in_dim(
                        arr, start, microbatch_size, axis=0
                    ),
                    batch_pytree,
                )

            first_batch = _slice_microbatch(batch, 0)
            (loss_0, per_target_0), grad_0 = value_and_grad(
                param_loss_fn, has_aux=True
            )(params, first_batch)

            def _accumulate(i, carry):
                grad_sum, loss_sum, per_target_sum = carry
                micro_batch = _slice_microbatch(batch, i)
                (loss_i, per_target_i), grad_i = value_and_grad(
                    param_loss_fn, has_aux=True
                )(params, micro_batch)
                grad_sum = tree_map(jnp.add, grad_sum, grad_i)
                loss_sum = loss_sum + loss_i
                per_target_sum = tree_map(jnp.add, per_target_sum, per_target_i)
                return grad_sum, loss_sum, per_target_sum

            grad_sum, loss_sum, per_target_sum = lax.fori_loop(
                1, microbatch_count, _accumulate, (grad_0, loss_0, per_target_0)
            )
            inv = jnp.asarray(1.0 / microbatch_count, dtype=loss_sum.dtype)
            grad = tree_map(lambda x: x * inv, grad_sum)
            loss = loss_sum * inv
            per_target_loss = tree_map(lambda x: x * inv, per_target_sum)
            return (loss, per_target_loss), grad

        stacked_microbatch_count = jax.tree_util.tree_leaves(batch)[0].shape[0]
        if stacked_microbatch_count != microbatch_count:
            raise ValueError(
                "Stacked microbatch axis mismatch for stack_scan mode. "
                f"Got stacked_microbatch_count={stacked_microbatch_count}, "
                f"microbatch_count={microbatch_count}"
            )

        first_batch = tree_map(lambda arr: arr[0], batch)
        if emit_shape_trace:
            stacked_shape = _shape_vec(jax.tree_util.tree_leaves(batch)[0])
            first_shape = _shape_vec(jax.tree_util.tree_leaves(first_batch)[0])
            jax.debug.print(
                "[ShapeTrace][micro] mode=stack_scan stacked_leaf_shape={} first_micro_leaf_shape={}",
                stacked_shape,
                first_shape,
            )
        (loss_0, per_target_0), grad_0 = value_and_grad(
            param_loss_fn, has_aux=True
        )(params, first_batch)
        grad_norm_0 = optax.global_norm(grad_0)

        def _scan_accumulate(carry, micro_batch):
            grad_sum, loss_sum, per_target_sum, grad_norm_sum, grad_norm_min, grad_norm_max = carry
            (loss_i, per_target_i), grad_i = value_and_grad(
                param_loss_fn, has_aux=True
            )(params, micro_batch)
            grad_sum = tree_map(jnp.add, grad_sum, grad_i)
            loss_sum = loss_sum + loss_i
            per_target_sum = tree_map(jnp.add, per_target_sum, per_target_i)
            grad_norm_i = optax.global_norm(grad_i)
            grad_norm_sum = grad_norm_sum + grad_norm_i
            grad_norm_min = jnp.minimum(grad_norm_min, grad_norm_i)
            grad_norm_max = jnp.maximum(grad_norm_max, grad_norm_i)
            return (
                grad_sum,
                loss_sum,
                per_target_sum,
                grad_norm_sum,
                grad_norm_min,
                grad_norm_max,
            ), None

        rest_batches = tree_map(lambda arr: arr[1:], batch)
        (grad_sum, loss_sum, per_target_sum, grad_norm_sum, grad_norm_min, grad_norm_max), _ = lax.scan(
            _scan_accumulate,
            (grad_0, loss_0, per_target_0, grad_norm_0, grad_norm_0, grad_norm_0),
            rest_batches,
            length=microbatch_count - 1,
        )
        inv = jnp.asarray(1.0 / microbatch_count, dtype=loss_sum.dtype)
        grad = tree_map(lambda x: x * inv, grad_sum)
        loss = loss_sum * inv
        per_target_loss = tree_map(lambda x: x * inv, per_target_sum)
        if debug_microbatch_grad_norms:
            jax.debug.print(
                "[GradAccumDebug] mode=stack_scan microbatch_count={} "
                "grad_norm_mean={:.6e} grad_norm_min={:.6e} grad_norm_max={:.6e}",
                microbatch_count,
                grad_norm_sum * inv,
                grad_norm_min,
                grad_norm_max,
            )
        return (loss, per_target_loss), grad

    def _accumulated_local_loss(params, batch, microbatch_count, accum_mode):
        if microbatch_count < 1:
            raise ValueError(f"microbatch_count must be >= 1, got {microbatch_count}")
        if microbatch_count == 1:
            return param_loss_fn(params, batch)

        if accum_mode == "concat_slice":
            local_batch_size = jax.tree_util.tree_leaves(batch)[0].shape[0]
            if local_batch_size % microbatch_count != 0:
                raise ValueError(
                    "Local batch size must be divisible by microbatch_count. "
                    f"Got local_batch_size={local_batch_size}, "
                    f"microbatch_count={microbatch_count}"
                )

            microbatch_size = local_batch_size // microbatch_count

            def _slice_microbatch(batch_pytree, i):
                start = i * microbatch_size
                return tree_map(
                    lambda arr: lax.dynamic_slice_in_dim(
                        arr, start, microbatch_size, axis=0
                    ),
                    batch_pytree,
                )

            first_batch = _slice_microbatch(batch, 0)
            loss_0, per_target_0 = param_loss_fn(params, first_batch)

            def _accumulate(i, carry):
                loss_sum, per_target_sum = carry
                micro_batch = _slice_microbatch(batch, i)
                loss_i, per_target_i = param_loss_fn(params, micro_batch)
                loss_sum = loss_sum + loss_i
                per_target_sum = tree_map(jnp.add, per_target_sum, per_target_i)
                return loss_sum, per_target_sum

            loss_sum, per_target_sum = lax.fori_loop(
                1, microbatch_count, _accumulate, (loss_0, per_target_0)
            )
            inv = jnp.asarray(1.0 / microbatch_count, dtype=loss_sum.dtype)
            loss = loss_sum * inv
            per_target_loss = tree_map(lambda x: x * inv, per_target_sum)
            return loss, per_target_loss

        stacked_microbatch_count = jax.tree_util.tree_leaves(batch)[0].shape[0]
        if stacked_microbatch_count != microbatch_count:
            raise ValueError(
                "Stacked microbatch axis mismatch for stack_scan mode. "
                f"Got stacked_microbatch_count={stacked_microbatch_count}, "
                f"microbatch_count={microbatch_count}"
            )

        first_batch = tree_map(lambda arr: arr[0], batch)
        loss_0, per_target_0 = param_loss_fn(params, first_batch)

        def _scan_accumulate(carry, micro_batch):
            loss_sum, per_target_sum = carry
            loss_i, per_target_i = param_loss_fn(params, micro_batch)
            loss_sum = loss_sum + loss_i
            per_target_sum = tree_map(jnp.add, per_target_sum, per_target_i)
            return (loss_sum, per_target_sum), None

        rest_batches = tree_map(lambda arr: arr[1:], batch)
        (loss_sum, per_target_sum), _ = lax.scan(
            _scan_accumulate,
            (loss_0, per_target_0),
            rest_batches,
            length=microbatch_count - 1,
        )
        inv = jnp.asarray(1.0 / microbatch_count, dtype=loss_sum.dtype)
        loss = loss_sum * inv
        per_target_loss = tree_map(lambda x: x * inv, per_target_sum)
        return loss, per_target_loss

    def _build_batch_update_fn(microbatch_count, accum_mode, emit_shape_trace=False):
        if microbatch_count < 1:
            raise ValueError(f"microbatch_count must be >= 1, got {microbatch_count}")
        batch_in_spec = _batch_in_spec(accum_mode, microbatch_count)

        def batch_update(params, opt_state, data):
            if mesh.size > 1:
                @partial(
                    shard_map,
                    mesh=mesh,
                    in_specs=batch_in_spec,
                    out_specs=PartitionSpec(),
                    check_rep=False,
                )
                def _inner(batch):
                    if emit_shape_trace:
                        local_shape = _shape_vec(jax.tree_util.tree_leaves(batch)[0])
                        jax.debug.print(
                            "[ShapeTrace][shard] accum_mode={} microbatch_count={} local_shard_leaf_shape={}",
                            accum_mode,
                            microbatch_count,
                            local_shape,
                        )
                    (loss, per_target_loss), grad = _accumulated_local_grad(
                        params,
                        batch,
                        microbatch_count,
                        accum_mode,
                        emit_shape_trace=emit_shape_trace,
                    )
                    # One synchronized reduction after accumulating local grads.
                    grad = _cast_tree_floating(grad, reduce_dtype)
                    loss = jnp.asarray(loss, dtype=reduce_dtype)
                    per_target_loss = _cast_tree_floating(per_target_loss, reduce_dtype)
                    grad = lax.pmean(grad, axis_name='batch')
                    loss = lax.pmean(loss, axis_name='batch')
                    per_target_loss = lax.pmean(per_target_loss, axis_name='batch')
                    grad = _cast_grad_like_params(grad, params)

                    new_params, new_opt_state = step_optimizer(
                        params, opt_state, grad, optimizer
                    )

                    return new_params, new_opt_state, loss, grad, per_target_loss
            else:
                def _inner(batch):
                    (loss, per_target_loss), grad = _accumulated_local_grad(
                        params,
                        batch,
                        microbatch_count,
                        accum_mode,
                        emit_shape_trace=emit_shape_trace,
                    )
                    new_params, new_opt_state = step_optimizer(
                        params, opt_state, grad, optimizer
                    )
                    return new_params, new_opt_state, loss, grad, per_target_loss

            return _inner(data)

        if enable_buffer_donation:
            donate_argnums = (
                donate_state_batch_argnums
                if donate_mode == "state_and_batch"
                else donate_state_argnums
            )
            return jit(batch_update, donate_argnums=donate_argnums)
        return jit(batch_update)

    def _get_batch_update_fn(microbatch_count, accum_mode, emit_shape_trace=False):
        key = (microbatch_count, accum_mode, bool(emit_shape_trace))
        batch_update = batch_update_fns.get(key)
        if batch_update is None:
            batch_update = _build_batch_update_fn(
                microbatch_count,
                accum_mode,
                emit_shape_trace=emit_shape_trace,
            )
            batch_update_fns[key] = batch_update
        return batch_update

    def _build_component_fns(microbatch_count, accum_mode):
        if microbatch_count < 1:
            raise ValueError(f"microbatch_count must be >= 1, got {microbatch_count}")
        batch_in_spec = _batch_in_spec(accum_mode, microbatch_count)

        if mesh.size > 1:
            def _add_device_axis(x):
                return jnp.expand_dims(x, axis=0)

            def _remove_device_axis(x):
                return jnp.squeeze(x, axis=0)

            @partial(
                shard_map,
                mesh=mesh,
                in_specs=(PartitionSpec(), batch_in_spec),
                # Keep local outputs sharded to avoid requiring inferred replication.
                out_specs=(
                    PartitionSpec('batch'),
                    PartitionSpec('batch'),
                    PartitionSpec('batch'),
                ),
                check_rep=False,
            )
            def local_grad_fn(params, batch):
                (loss, per_target_loss), grad = _accumulated_local_grad(
                    params, batch, microbatch_count, accum_mode
                )
                return (
                    _add_device_axis(loss),
                    tree_map(_add_device_axis, per_target_loss),
                    tree_map(_add_device_axis, grad),
                )

            @partial(
                shard_map,
                mesh=mesh,
                in_specs=(PartitionSpec(), batch_in_spec),
                out_specs=(PartitionSpec('batch'), PartitionSpec('batch')),
                check_rep=False,
            )
            def local_loss_fn(params, batch):
                loss, per_target_loss = _accumulated_local_loss(
                    params, batch, microbatch_count, accum_mode
                )
                return (
                    _add_device_axis(loss),
                    tree_map(_add_device_axis, per_target_loss),
                )

            @partial(
                shard_map,
                mesh=mesh,
                in_specs=(
                    PartitionSpec('batch'),
                    PartitionSpec('batch'),
                    PartitionSpec('batch'),
                ),
                out_specs=(PartitionSpec(), PartitionSpec(), PartitionSpec()),
                check_rep=False,
            )
            def collective_fn(loss, per_target_loss, grad):
                loss = jnp.asarray(loss, dtype=reduce_dtype)
                per_target_loss = _cast_tree_floating(per_target_loss, reduce_dtype)
                grad = _cast_tree_floating(grad, reduce_dtype)
                loss = lax.pmean(loss, axis_name='batch')
                per_target_loss = lax.pmean(per_target_loss, axis_name='batch')
                grad = lax.pmean(grad, axis_name='batch')
                return (
                    _remove_device_axis(loss),
                    tree_map(_remove_device_axis, per_target_loss),
                    tree_map(_remove_device_axis, grad),
                )
        else:
            def local_loss_fn(params, batch):
                return _accumulated_local_loss(params, batch, microbatch_count, accum_mode)

            def local_grad_fn(params, batch):
                (loss, per_target_loss), grad = _accumulated_local_grad(
                    params, batch, microbatch_count, accum_mode
                )
                return loss, per_target_loss, grad

            def collective_fn(loss, per_target_loss, grad):
                loss = jnp.asarray(loss, dtype=reduce_dtype)
                per_target_loss = _cast_tree_floating(per_target_loss, reduce_dtype)
                grad = _cast_tree_floating(grad, reduce_dtype)
                return loss, per_target_loss, grad
            local_loss_fn = jit(local_loss_fn)
            local_grad_fn = jit(local_grad_fn)
            collective_fn = jit(collective_fn)

        def optimizer_fn(params, opt_state, grad):
            return step_optimizer(params, opt_state, grad, optimizer)
        if enable_buffer_donation:
            optimizer_fn = jit(optimizer_fn, donate_argnums=donate_state_argnums)
        else:
            optimizer_fn = jit(optimizer_fn)

        return local_loss_fn, local_grad_fn, collective_fn, optimizer_fn

    def _get_component_fns(microbatch_count, accum_mode):
        key = (microbatch_count, accum_mode)
        fns = component_fns.get(key)
        if fns is None:
            fns = _build_component_fns(microbatch_count, accum_mode)
            component_fns[key] = fns
        return fns

    def update_fn(
        params,
        opt_state,
        batch,
        per_target=False,
        microbatch_count=1,
        accum_mode=None,
    ):
        idx = profile_step[0]
        profile_step[0] += 1
        profile_this_step = (
            profile_internal
            and (not profile_rank0_only or profile_rank == 0)
            and (profile_limit <= 0 or idx < profile_limit)
        )
        resolved_accum_mode = _resolve_accum_mode(accum_mode)
        if (
            compile_sig_debug
            and (not compile_sig_rank0_only or profile_rank == 0)
        ):
            signature = _shape_signature(batch, resolved_accum_mode, microbatch_count)
            if signature not in seen_shape_signatures:
                seen_shape_signatures.add(signature)
                leaf_shape = tuple(signature[2][0][0]) if signature[2] else ()
                leaf_dtype = signature[2][0][1] if signature[2] else "unknown"
                print(
                    "[UpdateFnShapeSignature] "
                    f"rank={profile_rank} step={idx} "
                    f"accum_mode={resolved_accum_mode} "
                    f"microbatch_count={microbatch_count} "
                    f"leaf0_shape={leaf_shape} leaf0_dtype={leaf_dtype} "
                    f"n_leaves={signature[3]} "
                    f"signature_hash={abs(hash(signature))}"
                )
        if profile_this_step:
            t_start = time.perf_counter()

        put_params_opt_ms = 0.0
        put_batch_ms = 0.0
        component_metrics = None
        if mesh.size > 1:
            if profile_this_step:
                t_put_params_opt_start = time.perf_counter()
            with jax.profiler.TraceAnnotation("chemtrain.update_fn.device_put_state"):
                params = device_put(params, replicate)
                opt_state = device_put(opt_state, replicate)
            if profile_this_step:
                t_put_params_opt_end = time.perf_counter()
                put_params_opt_ms = (
                    t_put_params_opt_end - t_put_params_opt_start
                ) * 1e3

            if profile_this_step:
                t_put_batch_start = time.perf_counter()
            with jax.profiler.TraceAnnotation("chemtrain.update_fn.device_put_batch"):
                batch = _put_process_local_data(
                    batch,
                    mesh,
                    _batch_in_spec(resolved_accum_mode, microbatch_count),
                )
            if profile_this_step:
                t_put_batch_end = time.perf_counter()
                put_batch_ms = (t_put_batch_end - t_put_batch_start) * 1e3

        block_loss_ms = float("nan")
        if profile_this_step and profile_components:
            local_loss_fn, local_grad_fn, collective_fn, optimizer_fn = _get_component_fns(
                microbatch_count, resolved_accum_mode
            )

            t_dispatch_start = time.perf_counter()
            forward_probe_metrics = None

            if profile_local_split:
                with jax.profiler.TraceAnnotation(
                    "chemtrain.update_fn.local_forward_probe_dispatch"
                ):
                    t_forward_dispatch_start = time.perf_counter()
                    loss_forward, per_target_forward = local_loss_fn(params, batch)
                    t_forward_dispatch_end = time.perf_counter()
                jax.block_until_ready(loss_forward)
                jax.block_until_ready(per_target_forward)
                t_forward_ready = time.perf_counter()
                forward_probe_metrics = {
                    "local_forward_dispatch_ms": (
                        t_forward_dispatch_end - t_forward_dispatch_start
                    ) * 1e3,
                    "local_forward_block_ms": (
                        t_forward_ready - t_forward_dispatch_end
                    ) * 1e3,
                    "local_forward_total_ms": (
                        t_forward_ready - t_forward_dispatch_start
                    ) * 1e3,
                }

            with jax.profiler.TraceAnnotation(
                "chemtrain.update_fn.local_grad_dispatch"
            ):
                t_local_dispatch_start = time.perf_counter()
                loss_local, per_target_local, grad_local = local_grad_fn(params, batch)
                t_local_dispatch_end = time.perf_counter()
            jax.block_until_ready(loss_local)
            t_local_ready = time.perf_counter()

            with jax.profiler.TraceAnnotation(
                "chemtrain.update_fn.collective_dispatch"
            ):
                t_collective_dispatch_start = time.perf_counter()
                loss_sync, per_target_sync, grad_sync = collective_fn(
                    loss_local, per_target_local, grad_local
                )
                t_collective_dispatch_end = time.perf_counter()
            jax.block_until_ready(loss_sync)
            t_collective_ready = time.perf_counter()

            with jax.profiler.TraceAnnotation(
                "chemtrain.update_fn.optimizer_dispatch"
            ):
                t_optimizer_dispatch_start = time.perf_counter()
                new_params, new_opt_state = optimizer_fn(
                    params, opt_state, grad_sync
                )
                t_optimizer_dispatch_end = time.perf_counter()
            jax.block_until_ready(new_params)
            t_optimizer_ready = time.perf_counter()

            t_dispatch_end = t_optimizer_dispatch_end
            block_loss_ms = (t_optimizer_ready - t_dispatch_start) * 1e3

            component_metrics = {
                "local_grad_dispatch_ms": (
                    t_local_dispatch_end - t_local_dispatch_start
                ) * 1e3,
                "local_grad_block_ms": (t_local_ready - t_local_dispatch_end) * 1e3,
                "collective_dispatch_ms": (
                    t_collective_dispatch_end - t_collective_dispatch_start
                ) * 1e3,
                "collective_block_ms": (
                    t_collective_ready - t_collective_dispatch_end
                ) * 1e3,
                "optimizer_dispatch_ms": (
                    t_optimizer_dispatch_end - t_optimizer_dispatch_start
                ) * 1e3,
                "optimizer_block_ms": (
                    t_optimizer_ready - t_optimizer_dispatch_end
                ) * 1e3,
                "sync_total_ms": (t_optimizer_ready - t_dispatch_start) * 1e3,
            }
            if forward_probe_metrics is not None:
                component_metrics.update(forward_probe_metrics)

            result = (
                new_params,
                new_opt_state,
                loss_sync,
                grad_sync,
                per_target_sync,
            )
        else:
            if profile_this_step:
                t_dispatch_start = time.perf_counter()
            emit_shape_trace = False
            if debug_shape_trace and not debug_shape_trace_printed[0]:
                emit_shape_trace = True
            with jax.profiler.TraceAnnotation("chemtrain.update_fn.batch_update_dispatch"):
                result = _get_batch_update_fn(
                    microbatch_count,
                    resolved_accum_mode,
                    emit_shape_trace=emit_shape_trace,
                )(
                    params,
                    opt_state,
                    batch,
                )
            if emit_shape_trace:
                debug_shape_trace_printed[0] = True
            if profile_this_step:
                t_dispatch_end = time.perf_counter()

        *outs, per_target_loss = result

        if (
            profile_this_step
            and profile_block
            and len(outs) >= 3
            and not profile_components
        ):
            # Block on the scalar batch loss without copying to host memory.
            t_block_start = time.perf_counter()
            jax.block_until_ready(outs[2])
            t_block_end = time.perf_counter()
            block_loss_ms = (t_block_end - t_block_start) * 1e3

        if profile_this_step:
            t_end = time.perf_counter()
            dispatch_ms = (t_dispatch_end - t_dispatch_start) * 1e3
            if not printed_meta[0] and len(outs) >= 3:
                printed_meta[0] = True
                loss_leaf = outs[2]
                loss_shape = getattr(loss_leaf, "shape", None)
                loss_dtype = getattr(loss_leaf, "dtype", None)
                loss_sharding = getattr(loss_leaf, "sharding", None)
                print(
                    "[UpdateFnInternalMeta] "
                    f"rank={profile_rank} step={idx} mesh_size={mesh.size} "
                    f"microbatch_count={microbatch_count} "
                    f"accum_mode={resolved_accum_mode} "
                    f"loss_shape={loss_shape} loss_dtype={loss_dtype} "
                    f"loss_sharding={loss_sharding}"
                )
            print(
                "[UpdateFnInternal] "
                f"rank={profile_rank} step={idx} mesh_size={mesh.size} "
                f"microbatch_count={microbatch_count} "
                f"accum_mode={resolved_accum_mode} "
                f"put_state_ms={put_params_opt_ms:.3f} "
                f"put_batch_ms={put_batch_ms:.3f} "
                f"dispatch_ms={dispatch_ms:.3f} "
                f"block_loss_ms={block_loss_ms:.3f} "
                f"total_ms={(t_end - t_start) * 1e3:.3f}"
            )
            if component_metrics is not None:
                local_grad_total = (
                    component_metrics["local_grad_dispatch_ms"]
                    + component_metrics["local_grad_block_ms"]
                )
                collective_total = (
                    component_metrics["collective_dispatch_ms"]
                    + component_metrics["collective_block_ms"]
                )
                optimizer_total = (
                    component_metrics["optimizer_dispatch_ms"]
                    + component_metrics["optimizer_block_ms"]
                )
                split_suffix = ""
                if "local_forward_total_ms" in component_metrics:
                    local_backward_estimated = max(
                        local_grad_total - component_metrics["local_forward_total_ms"],
                        0.0,
                    )
                    split_suffix = (
                        f" local_forward_total_ms={component_metrics['local_forward_total_ms']:.3f} "
                        f"local_forward_dispatch_ms={component_metrics['local_forward_dispatch_ms']:.3f} "
                        f"local_forward_block_ms={component_metrics['local_forward_block_ms']:.3f} "
                        f"local_backward_estimated_ms={local_backward_estimated:.3f}"
                    )
                print(
                    "[UpdateFnComponents] "
                    f"rank={profile_rank} step={idx} mesh_size={mesh.size} "
                    f"microbatch_count={microbatch_count} "
                    f"accum_mode={resolved_accum_mode} "
                    f"local_grad_total_ms={local_grad_total:.3f} "
                    f"collective_total_ms={collective_total:.3f} "
                    f"optimizer_total_ms={optimizer_total:.3f} "
                    f"sync_total_ms={component_metrics['sync_total_ms']:.3f} "
                    f"local_grad_dispatch_ms={component_metrics['local_grad_dispatch_ms']:.3f} "
                    f"local_grad_block_ms={component_metrics['local_grad_block_ms']:.3f} "
                    f"collective_dispatch_ms={component_metrics['collective_dispatch_ms']:.3f} "
                    f"collective_block_ms={component_metrics['collective_block_ms']:.3f} "
                    f"optimizer_dispatch_ms={component_metrics['optimizer_dispatch_ms']:.3f} "
                    f"optimizer_block_ms={component_metrics['optimizer_block_ms']:.3f}"
                    f"{split_suffix}"
                )

        if per_target:
            return *outs, per_target_loss
        else:
            return outs

    return update_fn


def shmap_loss_fn(batched_model, loss_fn, penalty_fn=None):
    """Initializes a shmapped function for computing a loss.

    Usage:
        .. code-block :: python

            loss, per_target_losses = loss_fn(params, batch, per_target=True)


    Args:
        batched_model: A model with signature model(params, batch), which
            predicts a batch of outputs used in loss function.
        loss_fn: Loss function(predictions, targets) returning the scalar loss
            value for a batch.
        penalty_fn: A penalty function based on the model parameters.

    Returns:
        A function that computes the total loss and per-target loss
        contributions.
    """
    # loss as function of params and batch for optimization.
    mesh = Mesh(jax.devices(), axis_names=('batch'))
    replicate = NamedSharding(mesh, PartitionSpec())
    split = NamedSharding(mesh, PartitionSpec('batch'))

    param_loss_fn = _get_param_loss_fn(loss_fn, batched_model, penalty_fn)

    @jit
    def batch_update(params, data):
        if mesh.size > 1:
            @partial(shard_map, mesh=mesh, in_specs=PartitionSpec('batch'),
                     out_specs=PartitionSpec(), check_rep=False)
            def _inner(batch):
                loss, per_target_loss = param_loss_fn(params, *batch)

                loss = lax.pmean(loss, axis_name='batch')
                per_target_loss = lax.pmean(per_target_loss, axis_name='batch')

                return loss, per_target_loss

        else:
            def _inner(batch):
                loss, per_target_loss = param_loss_fn(params, *batch)
                return loss, per_target_loss

        return _inner(data)

    def loss_fn(params, batch, mask=None, per_target=False):
        data = batch, mask
        if mesh.size > 1:
            params = device_put(params, replicate)
            data = _put_process_local_data(data, mesh, PartitionSpec('batch'))

        *outs, per_target_loss = batch_update(params, data)

        if per_target:
            return *outs, per_target_loss
        else:
            return outs

    return loss_fn


def shmap_model(batched_model):
    """Initializes a shmapped function for evaluating the model.

    Usage:
        .. code-block :: python

            predictions = shmapped_model(params, batch)


    Args:
        batched_model: A model with signature model(params, batch), which
            predicts a batch of outputs.

    Returns:
        A function that computes multiple predictions in parallel.

    """
    # loss as function of params and batch for optimization.
    mesh = Mesh(jax.devices(), axis_names=('batch'))
    replicate = NamedSharding(mesh, PartitionSpec())
    split = NamedSharding(mesh, PartitionSpec('batch'))

    @jit
    def batch_update(params, data):
        if mesh.size > 1:
            _inner = shard_map(
                batched_model, mesh=mesh,
                in_specs=(PartitionSpec(), PartitionSpec('batch')),
                out_specs=PartitionSpec('batch'),
                check_rep=False,
            )
        else:
            _inner = batched_model
        return _inner(params, data)

    def shmapped_model(params, batch):
        if mesh.size > 1:
            params = device_put(params, replicate)
            batch = _put_process_local_data(batch, mesh, PartitionSpec('batch'))

        return batch_update(params, batch)

    return shmapped_model


def init_val_predictions(batched_model, val_loader, batch_size=1,
                         batch_cache=10):
    """Model predictions for whole validation/test dataset.

    Usage:
        .. code-block :: python

            predictions, data_state = mapped_model_fn(params, data_state)

    Params needs to be N_devices times duplicated along axis 0.

    Args:
        batched_model: A model with signature model(params, batch), which
                       predicts a batch of outputs used in loss function.
        val_loader: Validation or test set NumpyDataLoader.
        batch_size: Total batch size that is processed in parallel
        batch_cache: Number of batches to cache.

    Returns:
        Tuple (predictions, data_state). predictions contains model predictions
        for the whole validation dataset and data_state is used to start the
        data loading in the next evaluation.
    """
    # case where validation data is very small
    batch_size = min(val_loader.static_information['observation_count']
                     // device_count(), batch_size)
    map_fun, data_release = data.full_data_mapper(val_loader, batch_cache,
                                                  batch_size)

    @jax.jit
    def single_batch(params, batch, unused_state):
        return batched_model(params, batch), unused_state

    def mapped_model_fn(params):
        params = jax.device_put(params, SingleDeviceSharding(jax.devices()[0]))
        predictions, _ = map_fun(partial(single_batch, params), None)
        return predictions
    return mapped_model_fn, data_release


def init_val_loss_fn(model, loss_fn, val_loader, val_targets_keys=None,
                     batch_size=1, batch_cache=100):
    """Initializes a pmapped loss function that computes the validation loss.

    Usage:
        .. code-block :: python

            val_loss, data_state = batched_loss_fn(params, data_state)

    Params needs to be N_devices times duplicated along axis 0.

    Args:
        model: A model with signature model(params, batch), which predicts
               outputs used in loss function.
        loss_fn: Loss function(predictions, targets) returning the scalar loss
                 value for a batch.
        val_loader: NumpyDataLoader for validation set.
        val_targets_keys: Dict containing targets of whole val
        batch_size: Total batch size that is processed in parallel.
        batch_cache: Number of batches to cache on GPU to reduce host-device
                     communication.

    Returns:
        A pmapped function that returns the average validation loss.
    """

    # We compute the validation error over the whole dataset at once, because
    # otherwise it is non-trivial to compute the correct error for masked
    # batches with different number of masked targets without explicitly knowing
    # the mask in this function
    # If predictions and targets of the whole validation dataset does not fit
    # memory, a more specialized approach needs to be taken.

    if val_targets_keys is None:
        target_data = val_loader.reference_data
    else:
        target_data = {key: val_loader.reference_data[key]
                       for key in val_targets_keys}

    mapped_predictions_fn, data_release_fn = init_val_predictions(
        model, val_loader, batch_size, batch_cache)

    def mapped_loss_fn(params):
        predictions = mapped_predictions_fn(params)
        val_loss = loss_fn(predictions, target_data)
        return val_loss

    return mapped_loss_fn, data_release_fn


def _batch_masked_loss(per_sample_loss, mask=None):
    # We do not divide by the number of samples here to avoid nans for
    # completely masked batches
    if mask is None:
        return jnp.mean(per_sample_loss)
    else:
        per_sample_loss = jnp.moveaxis(per_sample_loss, 0, -1)
        return jnp.mean(per_sample_loss * mask)


def _masked_loss(per_element_loss, mask=None, weights=None):
    """Computes average loss, accounting for masked elements, if applicable."""
    if weights is not None:
        if per_element_loss.ndim > 0:
            per_element_loss = jnp.moveaxis(per_element_loss, 0, -1)
            per_element_loss *= weights
            per_element_loss = jnp.moveaxis(per_element_loss, -1, 0)
        else:
            per_element_loss *= weights

    if mask is None:
        return jnp.mean(per_element_loss)
    else:
        assert mask.shape == per_element_loss.shape, (
            'Mask requires same shape as targets.'
        )
        return jnp.sum(per_element_loss * mask) / jnp.sum(mask)


def mse_loss(predictions, targets, mask=None, weights=None):
    """Computes mean squared error loss for given predictions and targets.

    Args:
        predictions: Array of predictions
        targets: Array of respective targets. Needs to have same shape as
                 predictions.
        mask: Mask contribution of some array elements. Needs to have same shape
              as predictions. Default None applies no mask.

    Returns:
        Mean squared error loss value.
    """
    squared_differences = jnp.square(targets - predictions)
    return _masked_loss(squared_differences, mask, weights)


def mae_loss(predictions, targets, mask=None, weights=None):
    """Computes the mean absolute error for given predictions and targets.

    Args:
        predictions: Array of predictions
        targets: Array of respective targets. Needs to have same shape as
                 predictions.
        mask: Mask contribution of some array elements. Needs to have same shape
              as predictions. Default None applies no mask.

    Returns:
        Mean absolute error value.
    """

    # Set gradients to zero at singularity
    safe_mask = (targets - predictions) != 0.0
    safe_diff = jnp.where(safe_mask, targets - predictions, 1.0)
    abs_err = jnp.abs(safe_diff) * safe_mask
    return _masked_loss(abs_err, mask, weights)


def identity_loss(predictions, *args, **kwargs):
    """Considers the prediction itself as loss value.

    For example, the relative entropy can be used directly as loss in DiffTRe.

    Args:
        predictions: Array of predictions (scalar)

    Returns:
        Returns the prediction itself as loss value.

    """
    del args, kwargs
    return predictions


def step_optimizer(params, opt_state, grad, optimizer):
    """Steps optimizer and updates state using the gradient."""
    grad = _cast_grad_like_params(grad, params)
    scaled_grad, new_opt_state = optimizer.update(grad, opt_state, params)
    new_params = optax.apply_updates(params, scaled_grad)
    return new_params, new_opt_state
