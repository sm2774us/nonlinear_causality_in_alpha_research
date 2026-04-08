# AlphaPod v2 Model Artefacts

Place compiled MAX Engine model files here.

## v2 Model (`alpha_macro_v2.max`)
- Input: `(N, 9)` float32 — [f_tsmom, f_carry, f_vol, f_tsmom_x_vol, f_carry_x_vol,
  f_regime_mom, f_vol_regime_adj, f_te_net_cause, f_ccm_driver]
- Output: `(N,)` float32 — raw alpha scores
- Architecture: GBM or Transformer with walk-forward validation

## v1 Model (`alpha_macro.max`) — backward compatibility
- Input: `(N, 3)` float32 — [f_tsmom, f_carry, f_vol]

## Compile
```bash
max build --input-spec "input[N,9]float32" model.onnx -o alpha_macro_v2.max
```
