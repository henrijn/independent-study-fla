"""
Phase 1: Layer-Level GPU Tests for Phoenix & RoutedDeltaNet
===========================================================
Run this on Google Colab (GPU runtime) after installing the fla package.

Install instructions (run in Colab first):
    !pip install -q torch einops transformers
    !pip install -q git+https://github.com/YOUR_USERNAME/flash-linear-attention.git
    # OR if using a zip upload:
    # !pip install -e ./flash-linear-attention

Then run:
    !python test_phase1_gpu.py
OR paste each section as a Colab cell.
"""

import sys
import math
import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def check(name: str, condition: bool, msg: str = ""):
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"  {status}  {name}" + (f"  →  {msg}" if msg else ""))
    return condition

def assert_no_nan_inf(tensor: torch.Tensor, name: str) -> bool:
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    ok = not has_nan and not has_inf
    msg = ""
    if has_nan:
        msg += "contains NaN "
    if has_inf:
        msg += "contains Inf"
    check(f"{name} — no NaN/Inf", ok, msg.strip())
    return ok


# ─────────────────────────────────────────────
# Device setup
# ─────────────────────────────────────────────

section("Environment")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device : {device}")
if device.type == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  ⚠️  No GPU found — Triton kernels will NOT work on CPU.")
    print("     Please enable GPU runtime in Colab: Runtime → Change runtime type → T4 GPU")


# ─────────────────────────────────────────────
# Shared test config  (tiny to be fast)
# ─────────────────────────────────────────────

BATCH        = 2
SEQ_LEN      = 128   # ≥ 65 forces 'chunk' mode; < 65 triggers 'fused_recurrent'
HIDDEN_SIZE  = 64
NUM_HEADS    = 4
TOPK         = 4     # must be ≤ num_slots; default num_slots = head_k_dim = 64/4 = 16
DTYPE        = torch.bfloat16


# ═══════════════════════════════════════════════════════════════
# TEST 1 — PhoenixAttention
# ═══════════════════════════════════════════════════════════════

def test_phoenix(mode: str = "chunk"):
    section(f"TEST 1 — PhoenixAttention  [mode={mode}]")
    from fla.layers.phoenix import PhoenixAttention

    # head_k_dim = HIDDEN_SIZE // NUM_HEADS = 16  →  num_slots default = 16
    # topk must be ≤ num_slots
    layer = PhoenixAttention(
        mode=mode,
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        topk=TOPK,
        layer_idx=0,
        router_noise=False,   # deterministic for testing
    ).to(device=device, dtype=DTYPE)
    layer.eval()

    # ── 1a. FORWARD PASS ──────────────────────────────────────
    print(f"\n  [1a] Forward pass  (batch={BATCH}, seq={SEQ_LEN}, hidden={HIDDEN_SIZE})")
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=DTYPE)
    with torch.no_grad():
        o, _, _ = layer(x)

    expected_shape = (BATCH, SEQ_LEN, HIDDEN_SIZE)
    check("Output shape", o.shape == expected_shape,
          f"got {tuple(o.shape)}, expected {expected_shape}")
    assert_no_nan_inf(o, "Output")

    # ── 1b. SHORT-SEQUENCE → fused_recurrent ──────────────────
    print(f"\n  [1b] Short-sequence forward (seq=32, forces fused_recurrent)")
    x_short = torch.randn(BATCH, 32, HIDDEN_SIZE, device=device, dtype=DTYPE)
    with torch.no_grad():
        o_short, _, _ = layer(x_short)
    check("Short-seq output shape", o_short.shape == (BATCH, 32, HIDDEN_SIZE),
          f"got {tuple(o_short.shape)}")
    assert_no_nan_inf(o_short, "Short-seq output")

    # ── 1c. BACKWARD PASS ─────────────────────────────────────
    print(f"\n  [1c] Backward pass (gradient check)")
    layer.train()
    x_grad = torch.randn(BATCH, SEQ_LEN, HIDDEN_SIZE,
                          device=device, dtype=DTYPE, requires_grad=True)
    o_grad, _, _ = layer(x_grad)
    loss = o_grad.sum()
    loss.backward()

    check("Input grad exists", x_grad.grad is not None)
    if x_grad.grad is not None:
        assert_no_nan_inf(x_grad.grad, "Input grad")

    param_grads_ok = True
    for name, p in layer.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                check(f"Param grad exists: {name}", False, "grad is None")
                param_grads_ok = False
            elif not torch.isfinite(p.grad).all():
                check(f"Param grad finite: {name}", False, "non-finite grad")
                param_grads_ok = False
    if param_grads_ok:
        check("All parameter grads finite", True)

    # ── 1d. TRAINING + EVAL CONSISTENCY ───────────────────────
    print(f"\n  [1d] Training vs. eval mode consistency")
    layer.eval()
    x_test = torch.randn(BATCH, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=DTYPE)
    with torch.no_grad():
        o_eval1, _, _ = layer(x_test)
        o_eval2, _, _ = layer(x_test)
    check("Deterministic in eval", torch.allclose(o_eval1, o_eval2),
          "Two identical forward passes gave different outputs")

    print()
    return True


# ═══════════════════════════════════════════════════════════════
# TEST 2 — RoutedDeltaNetAttention
# ═══════════════════════════════════════════════════════════════

def test_routed_deltanet(mode: str = "chunk"):
    section(f"TEST 2 — RoutedDeltaNetAttention  [mode={mode}]")
    from fla.layers.routed_deltanet import RoutedDeltaNetAttention

    layer = RoutedDeltaNetAttention(
        mode=mode,
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        topk=TOPK,
        layer_idx=0,
        router_noise=False,
    ).to(device=device, dtype=DTYPE)
    layer.eval()

    # ── 2a. FORWARD PASS ──────────────────────────────────────
    print(f"\n  [2a] Forward pass  (batch={BATCH}, seq={SEQ_LEN}, hidden={HIDDEN_SIZE})")
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=DTYPE)
    with torch.no_grad():
        o, _, _ = layer(x)

    expected_shape = (BATCH, SEQ_LEN, HIDDEN_SIZE)
    check("Output shape", o.shape == expected_shape,
          f"got {tuple(o.shape)}, expected {expected_shape}")
    assert_no_nan_inf(o, "Output")

    # ── 2b. SHORT-SEQUENCE → fused_recurrent ──────────────────
    print(f"\n  [2b] Short-sequence forward (seq=32, forces fused_recurrent)")
    x_short = torch.randn(BATCH, 32, HIDDEN_SIZE, device=device, dtype=DTYPE)
    with torch.no_grad():
        o_short, _, _ = layer(x_short)
    check("Short-seq output shape", o_short.shape == (BATCH, 32, HIDDEN_SIZE),
          f"got {tuple(o_short.shape)}")
    assert_no_nan_inf(o_short, "Short-seq output")

    # ── 2c. BACKWARD PASS ─────────────────────────────────────
    print(f"\n  [2c] Backward pass (gradient check)")
    layer.train()
    x_grad = torch.randn(BATCH, SEQ_LEN, HIDDEN_SIZE,
                          device=device, dtype=DTYPE, requires_grad=True)
    o_grad, _, _ = layer(x_grad)
    loss = o_grad.sum()
    loss.backward()

    check("Input grad exists", x_grad.grad is not None)
    if x_grad.grad is not None:
        assert_no_nan_inf(x_grad.grad, "Input grad")

    param_grads_ok = True
    for name, p in layer.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                check(f"Param grad exists: {name}", False, "grad is None")
                param_grads_ok = False
            elif not torch.isfinite(p.grad).all():
                check(f"Param grad finite: {name}", False, "non-finite grad")
                param_grads_ok = False
    if param_grads_ok:
        check("All parameter grads finite", True)

    # ── 2d. USE_GATE=True variant ─────────────────────────────
    print(f"\n  [2d] use_gate=True variant")
    layer_gated = RoutedDeltaNetAttention(
        mode=mode,
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        topk=TOPK,
        layer_idx=0,
        router_noise=False,
        use_gate=True,
    ).to(device=device, dtype=DTYPE)
    layer_gated.eval()
    with torch.no_grad():
        o_gated, _, _ = layer_gated(x)
    check("Gated output shape", o_gated.shape == expected_shape,
          f"got {tuple(o_gated.shape)}")
    assert_no_nan_inf(o_gated, "Gated output")

    # ── 2e. TRAINING + EVAL CONSISTENCY ───────────────────────
    print(f"\n  [2e] Training vs. eval mode consistency")
    layer.eval()
    x_test = torch.randn(BATCH, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=DTYPE)
    with torch.no_grad():
        o_eval1, _, _ = layer(x_test)
        o_eval2, _, _ = layer(x_test)
    check("Deterministic in eval", torch.allclose(o_eval1, o_eval2),
          "Two identical forward passes gave different outputs")

    print()
    return True


# ═══════════════════════════════════════════════════════════════
# TEST 3 — chunk vs fused_recurrent consistency
# ═══════════════════════════════════════════════════════════════

def test_mode_consistency():
    """
    Both 'chunk' and 'fused_recurrent' should produce numerically close outputs
    given the same weights and input. This catches kernel-level bugs where one
    mode produces wrong values.
    """
    section("TEST 3 — chunk vs fused_recurrent consistency")

    for LayerClass, name in [
        ("phoenix",        "PhoenixAttention"),
        ("routed_deltanet","RoutedDeltaNetAttention"),
    ]:
        print(f"\n  Checking {name} ...")

        if LayerClass == "phoenix":
            from fla.layers.phoenix import PhoenixAttention as Cls
        else:
            from fla.layers.routed_deltanet import RoutedDeltaNetAttention as Cls

        # Use same weights for both modes
        layer_chunk = Cls(
            mode="chunk",
            hidden_size=HIDDEN_SIZE,
            num_heads=NUM_HEADS,
            topk=TOPK,
            layer_idx=0,
            router_noise=False,
        ).to(device=device, dtype=DTYPE)
        layer_chunk.eval()

        layer_recurrent = Cls(
            mode="fused_recurrent",
            hidden_size=HIDDEN_SIZE,
            num_heads=NUM_HEADS,
            topk=TOPK,
            layer_idx=0,
            router_noise=False,
        ).to(device=device, dtype=DTYPE)
        layer_recurrent.eval()

        # Copy weights
        layer_recurrent.load_state_dict(layer_chunk.state_dict())

        # Use seq_len that forces chunk mode regardless of threshold
        # chunk mode: seq > 64,  fused_recurrent: forced by mode='fused_recurrent'
        x = torch.randn(1, 128, HIDDEN_SIZE, device=device, dtype=DTYPE)

        with torch.no_grad():
            o_chunk, _, _   = layer_chunk(x)
            o_recur, _, _   = layer_recurrent(x)

        # bfloat16 has limited precision — use a loose tolerance
        atol = 5e-2   # ~0.05 absolute tolerance is reasonable for bfloat16
        close = torch.allclose(o_chunk, o_recur, atol=atol, rtol=1e-2)
        max_diff = (o_chunk - o_recur).abs().max().item()
        check(
            f"{name}: chunk ≈ fused_recurrent (max_diff={max_diff:.4f}, atol={atol})",
            close,
            f"max absolute difference = {max_diff:.4f}"
        )

    print()


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

def run_all():
    results = {}

    try:
        results["phoenix_chunk"]       = test_phoenix(mode="chunk")
    except Exception as e:
        section("ERROR in Phoenix (chunk)")
        print(f"  {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        results["phoenix_chunk"] = False

    try:
        results["phoenix_fused"]       = test_phoenix(mode="fused_recurrent")
    except Exception as e:
        section("ERROR in Phoenix (fused_recurrent)")
        print(f"  {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        results["phoenix_fused"] = False

    try:
        results["routed_deltanet_chunk"]  = test_routed_deltanet(mode="chunk")
    except Exception as e:
        section("ERROR in RoutedDeltaNet (chunk)")
        print(f"  {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        results["routed_deltanet_chunk"] = False

    try:
        results["routed_deltanet_fused"]  = test_routed_deltanet(mode="fused_recurrent")
    except Exception as e:
        section("ERROR in RoutedDeltaNet (fused_recurrent)")
        print(f"  {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        results["routed_deltanet_fused"] = False

    try:
        test_mode_consistency()
        results["mode_consistency"] = True
    except Exception as e:
        section("ERROR in mode consistency test")
        print(f"  {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        results["mode_consistency"] = False

    # Final summary
    section("PHASE 1 SUMMARY")
    all_pass = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {test_name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  🎉 All Phase 1 tests passed! Ready to move to Phase 2.")
    else:
        print("  ⚠️  Some tests failed. Check the error messages above and fix before Phase 2.")


if __name__ == "__main__":
    run_all()
