#!/usr/bin/env python3
"""v40: give Inkling the EAGLE3 aux-hidden-state interface so DSpark can drive it.

The RadixArk/Inkling-Small-DSpark drafter needs auxiliary hidden states from target
layers [1,6,12,17,23,28,34,39]. vLLM gates that behind the SupportsEagle3 protocol,
and Inkling's model class does not implement it:

    RuntimeError: Model does not support EAGLE3 interface

This adds:
  1. InklingModel.aux_hidden_state_layers  (default empty -> zero overhead when unused)
  2. aux capture inside the decoder-layer loop
  3. SupportsEagle3 + the two accessor methods on _TmlForCausalLMBase

Caveat worth knowing: Inkling defers part of each layer's residual/sconv work via
`pending`, so the value captured after layer i does not yet include that layer's
deferred contribution. This is an approximation of the hidden state the drafter was
trained on (SGLang). It can only reduce ACCEPTANCE, never correctness - speculative
tokens are always verified against the target. If acceptance is poor, this is the
first thing to revisit.

Run inside the container:
    python3 /tmp/patch_eagle3_iface_v40.py
"""
import pathlib, sys

MODEL = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/inkling/nvidia/model.py")


def main() -> int:
    s = MODEL.read_text()
    if "aux_hidden_state_layers" in s:
        print("v40: already patched")
        return 0
    orig = s

    # --- 1. import the protocol -------------------------------------------------
    if "SupportsEagle3" not in s:
        paren = "from vllm.model_executor.models.interfaces import ("
        plain = "from vllm.model_executor.models.interfaces import "
        if paren in s:
            # multi-line parenthesised import: add inside the parens
            s = s.replace(paren, paren + "\n    SupportsEagle3,", 1)
        elif plain in s:
            s = s.replace(plain, plain + "SupportsEagle3, ", 1)
        else:
            s = s.replace(
                "class InklingDecoderLayer(nn.Module):",
                "from vllm.model_executor.models.interfaces import SupportsEagle3\n\n\n"
                "class InklingDecoderLayer(nn.Module):", 1)

    # --- 2. declare the attribute on InklingModel -------------------------------
    anchor = ('        self.norm = InklingRMSNorm(config.hidden_size, '
              'eps=config.rms_norm_eps)')
    assert anchor in s, "InklingModel.norm anchor not found"
    s = s.replace(anchor,
                  "        # EAGLE3/DSpark aux hidden states; empty tuple = feature off.\n"
                  "        self.aux_hidden_state_layers: tuple[int, ...] = ()\n" + anchor, 1)

    # --- 3. capture aux states in the layer loop --------------------------------
    old_loop = """        pending: tuple[InklingDelta, InklingShortConv] | None = None
        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, pending = layer(
                positions,
                hidden_states,
                pending=pending,
                defer_mlp_add=True,
                attn_in=attn_in0,
                log_scaling=log_scaling,
            )
            attn_in0 = None
"""
    new_loop = """        pending: tuple[InklingDelta, InklingShortConv] | None = None
        _aux_layers = self.aux_hidden_state_layers
        _aux_out: list[torch.Tensor] = []
        for _off, layer in enumerate(self.layers[self.start_layer : self.end_layer]):
            hidden_states, pending = layer(
                positions,
                hidden_states,
                pending=pending,
                defer_mlp_add=True,
                attn_in=attn_in0,
                log_scaling=log_scaling,
            )
            attn_in0 = None
            if _aux_layers and (self.start_layer + _off) in _aux_layers:
                _aux_out.append(hidden_states)
"""
    assert old_loop in s, "decoder layer loop not found"
    s = s.replace(old_loop, new_loop, 1)

    # --- 4. return aux states alongside the final hidden state -------------------
    old_tail = """        if pending is not None:
            # Final RS/sconv/AG + residual add fused with the final rmsnorm.
            norm_out = _sconv_add_norm(
                pending[0], hidden_states, pending[1], self.norm, positions
            )[0]
            assert norm_out is not None
            return norm_out
        return self.norm(hidden_states)"""
    new_tail = """        if pending is not None:
            # Final RS/sconv/AG + residual add fused with the final rmsnorm.
            norm_out = _sconv_add_norm(
                pending[0], hidden_states, pending[1], self.norm, positions
            )[0]
            assert norm_out is not None
            return (norm_out, _aux_out) if _aux_layers else norm_out
        _final = self.norm(hidden_states)
        return (_final, _aux_out) if _aux_layers else _final"""
    assert old_tail in s, "forward tail not found"
    s = s.replace(old_tail, new_tail, 1)

    # --- 5. implement the protocol on the causal-LM base ------------------------
    old_cls = "class _TmlForCausalLMBase(nn.Module, SupportsPP, SupportsLoRA):"
    new_cls = ("class _TmlForCausalLMBase(nn.Module, SupportsPP, SupportsLoRA, "
               "SupportsEagle3):")
    assert old_cls in s, "causal-LM base class not found"
    s = s.replace(old_cls, new_cls, 1)

    # insert the accessors right after the class docstring/first def
    marker = new_cls + "\n"
    idx = s.index(marker) + len(marker)
    accessors = (
        "    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:\n"
        "        self.model.aux_hidden_state_layers = tuple(layers)\n"
        "\n"
        "    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:\n"
        "        return self.model.aux_hidden_state_layers\n"
        "\n"
        "    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:\n"
        "        n = len(self.model.layers)\n"
        "        return (2, n // 2, max(n - 3, 0))\n"
        "\n"
    )
    s = s[:idx] + accessors + s[idx:]

    if s == orig:
        print("v40: NOTHING CHANGED", file=sys.stderr)
        return 1
    MODEL.write_text(s)
    print("v40: patched EAGLE3 interface into Inkling model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
