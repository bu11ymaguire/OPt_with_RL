# Stage 2 결과표

`scripts/make_report.py` 가 raw 결과에서 생성한다. **손으로 수정하지 않는다.**

## 통계 표기 규칙 (프로토콜 D26)

```text
p 를 0.0000 으로 쓰지 않는다. 부트스트랩/순열 기반이므로 p<0.0001 로 쓴다
equivalence margin 을 사전 등록하지 않았으므로 "효과가 0 이다" 를 주장하지 않는다
CI 가 좁게 0 을 포함하면 "실용적으로 큰 이득이 관측되지 않았다" 로 쓴다
```

## beam 4 dev pilot

원래 dev subset 3 spec x 3 seed. 2개 spec 이 포화됐다 (D19)

```text
raw        headroom_pilot_step_size_fixed_b4_9d725689.jsonl
완료 run   324
best_open_loop -> open_loop[7]
best_static -> static[7]
```

절대 median logΔ (nat, 높을수록 좋다)

| controller | quad_ill_conditioned_d100_k1e+05 | quad_spd_d64_k1e+02 | rosen_d2_s100_std |
|---|---|---|---|
| `best_static` | 9.337 (n=3) | 31.438 (n=3) | 7.092 (n=3) |
| `best_open_loop` | 9.229 (n=3) | 25.456 (n=3) | 31.438 (n=3) |
| `heuristic` | 8.566 (n=3) | 31.438 (n=3) | 31.438 (n=3) |
| `onestep_narrow` | 9.385 (n=3) | 31.438 (n=3) | 1.670 (n=3) |
| `onestep_absolute` | 9.397 (n=3) | 31.438 (n=3) | 31.438 (n=3) |
| `committed_Q4_narrow` | 9.601 (n=3) | 31.438 (n=3) | 1.670 (n=3) |
| `shrinking_Q4_narrow` | 9.831 (n=3) | 31.438 (n=3) | 1.670 (n=3) |

paired delta (nat, 양수면 treatment 가 좋다)

| 비교 | 범위 | median | 95% CI | p | n | 양수 |
|---|---|---|---|---|---|---|
| A2 `shrinking_Q4_narrow` − `best_static` | ALL | +0.000 | [-5.422, +0.494] | 0.4375 | 9 | 3/9 |
| A2 `shrinking_Q4_narrow` − `best_static` | quad_ill_conditioned_d100_k1e+05 | +0.494 | [+0.072, +0.737] | n/a | 3 | 3/3 |
| A2 `shrinking_Q4_narrow` − `best_static` | quad_spd_d64_k1e+02 | +0.000 | [+0.000, +0.000] | n/a | 3 | 0/3 |
| A2 `shrinking_Q4_narrow` − `best_static` | rosen_d2_s100_std | -5.422 | [-5.422, -5.422] | n/a | 3 | 0/3 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | ALL | +0.000 | [+0.000, +0.542] | n/a | 9 | 2/9 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | quad_ill_conditioned_d100_k1e+05 | +0.542 | [-0.191, +0.873] | n/a | 3 | 2/3 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | quad_spd_d64_k1e+02 | +0.000 | [+0.000, +0.000] | n/a | 3 | 0/3 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | rosen_d2_s100_std | +0.000 | [+0.000, +0.000] | n/a | 3 | 0/3 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | ALL | +0.000 | [+0.000, +0.370] | n/a | 9 | 3/9 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | quad_ill_conditioned_d100_k1e+05 | +0.370 | [+0.230, +0.520] | n/a | 3 | 3/3 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | quad_spd_d64_k1e+02 | +0.000 | [+0.000, +0.000] | n/a | 3 | 0/3 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | rosen_d2_s100_std | +0.000 | [+0.000, +0.000] | n/a | 3 | 0/3 |
| open_loop `best_open_loop` − `best_static` | ALL | -0.137 | [-5.982, +24.346] | 0.9062 | 9 | 3/9 |
| open_loop `best_open_loop` − `best_static` | quad_ill_conditioned_d100_k1e+05 | -0.137 | [-0.184, -0.108] | n/a | 3 | 0/3 |
| open_loop `best_open_loop` − `best_static` | quad_spd_d64_k1e+02 | -5.982 | [-6.324, -5.562] | n/a | 3 | 0/3 |
| open_loop `best_open_loop` − `best_static` | rosen_d2_s100_std | +24.346 | [+24.346, +24.346] | n/a | 3 | 3/3 |

탐색 비용과 거절률 (전체 인스턴스 median)

| controller | decision-search GE | object GE | 거절률 |
|---|---|---|---|
| `best_static` | 0 | 133.8 | 0.00 |
| `best_open_loop` | 0 | 133.4 | 0.00 |
| `heuristic` | 0 | 147.6 | 0.00 |
| `onestep_narrow` | 1,001 | 139.1 | 0.00 |
| `onestep_absolute` | 6,900 | 139.1 | 0.00 |
| `committed_Q4_narrow` | 50,236 | 143.8 | 0.00 |
| `shrinking_Q4_narrow` | 196,675 | 147.1 | 0.00 |

## beam 8 challenge dev

challenge 4 spec x seeds 2/3/4. 설정 선택에 사용 (D21/D22)

```text
raw        headroom_challenge_step_size_fixed_b8_fed9aebd.jsonl
완료 run   360
best_open_loop -> open_loop[4]
best_static -> static[6]
```

절대 median logΔ (nat, 높을수록 좋다)

| controller | quad_ill_conditioned_d100_k1e+03 | quad_ill_conditioned_d100_k1e+04 | quad_ill_conditioned_d100_k1e+05 | quad_ill_conditioned_d100_k1e+06 |
|---|---|---|---|---|
| `best_static` | 13.110 (n=3) | 8.586 (n=3) | 8.151 (n=3) | 9.051 (n=3) |
| `best_open_loop` | 14.368 (n=3) | 8.914 (n=3) | 8.571 (n=3) | 9.500 (n=3) |
| `heuristic` | 13.109 (n=3) | 8.586 (n=3) | 8.151 (n=3) | 9.033 (n=3) |
| `onestep_narrow` | 19.605 (n=3) | 10.494 (n=3) | 9.385 (n=3) | 10.116 (n=3) |
| `onestep_absolute` | 19.622 (n=3) | 9.782 (n=3) | 9.397 (n=3) | 10.143 (n=3) |
| `committed_Q4_narrow` | 19.536 (n=3) | 10.974 (n=3) | 9.604 (n=3) | 10.433 (n=3) |
| `shrinking_Q4_narrow` | 20.002 (n=3) | 11.062 (n=3) | 9.474 (n=3) | 10.530 (n=3) |

paired delta (nat, 양수면 treatment 가 좋다)

| 비교 | 범위 | median | 95% CI | p | n | 양수 |
|---|---|---|---|---|---|---|
| A2 `shrinking_Q4_narrow` − `best_static` | ALL | +1.502 | [+1.160, +4.402] | 0.0005 | 12 | 12/12 |
| A2 `shrinking_Q4_narrow` − `best_static` | quad_ill_conditioned_d100_k1e+03 | +6.584 | [+6.329, +7.152] | n/a | 3 | 3/3 |
| A2 `shrinking_Q4_narrow` − `best_static` | quad_ill_conditioned_d100_k1e+04 | +2.361 | [+1.523, +2.476] | n/a | 3 | 3/3 |
| A2 `shrinking_Q4_narrow` − `best_static` | quad_ill_conditioned_d100_k1e+05 | +0.735 | [+0.605, +1.464] | n/a | 3 | 3/3 |
| A2 `shrinking_Q4_narrow` − `best_static` | quad_ill_conditioned_d100_k1e+06 | +1.232 | [+1.087, +1.480] | n/a | 3 | 3/3 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | ALL | +0.251 | [-0.036, +0.487] | 0.0771 | 12 | 8/12 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | quad_ill_conditioned_d100_k1e+03 | +0.299 | [+0.193, +0.657] | n/a | 3 | 3/3 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | quad_ill_conditioned_d100_k1e+04 | +0.560 | [-0.468, +1.320] | n/a | 3 | 2/3 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | quad_ill_conditioned_d100_k1e+05 | -0.056 | [-0.199, +0.229] | n/a | 3 | 1/3 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | quad_ill_conditioned_d100_k1e+06 | +0.272 | [-0.016, +0.413] | n/a | 3 | 2/3 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | ALL | -0.044 | [-0.590, +0.070] | 0.3804 | 12 | 6/12 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | quad_ill_conditioned_d100_k1e+03 | +0.000 | [-1.428, +1.339] | n/a | 3 | 2/3 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | quad_ill_conditioned_d100_k1e+04 | -0.088 | [-0.627, +0.087] | n/a | 3 | 1/3 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | quad_ill_conditioned_d100_k1e+05 | -0.410 | [-0.553, +0.010] | n/a | 3 | 1/3 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | quad_ill_conditioned_d100_k1e+06 | +0.053 | [-0.724, +0.096] | n/a | 3 | 2/3 |
| open_loop `best_open_loop` − `best_static` | ALL | +0.413 | [+0.317, +0.754] | 0.0005 | 12 | 12/12 |
| open_loop `best_open_loop` − `best_static` | quad_ill_conditioned_d100_k1e+03 | +1.120 | [+1.057, +1.258] | n/a | 3 | 3/3 |
| open_loop `best_open_loop` − `best_static` | quad_ill_conditioned_d100_k1e+04 | +0.405 | [+0.328, +0.451] | n/a | 3 | 3/3 |
| open_loop `best_open_loop` − `best_static` | quad_ill_conditioned_d100_k1e+05 | +0.317 | [+0.215, +0.420] | n/a | 3 | 3/3 |
| open_loop `best_open_loop` − `best_static` | quad_ill_conditioned_d100_k1e+06 | +0.316 | [+0.209, +0.449] | n/a | 3 | 3/3 |

탐색 비용과 거절률 (전체 인스턴스 median)

| controller | decision-search GE | object GE | 거절률 |
|---|---|---|---|
| `best_static` | 0 | 147.6 | 0.00 |
| `best_open_loop` | 0 | 144.5 | 0.00 |
| `heuristic` | 0 | 147.6 | 0.00 |
| `onestep_narrow` | 1,186 | 141.7 | 0.00 |
| `onestep_absolute` | 10,552 | 141.2 | 0.00 |
| `committed_Q4_narrow` | 69,336 | 143.7 | 0.00 |
| `shrinking_Q4_narrow` | 193,894 | 144.0 | 0.00 |

## beam 8 held-out confirmatory

같은 4 spec x seeds 100~109. 사전 고정 설정, 최종 효과 추정 (D26)

```text
raw        headroom_challenge-heldout_step_size_fixed_b8_9a18b6e9.jsonl
완료 run   960
best_open_loop -> open_loop[4]
best_static -> static[2]
```

절대 median logΔ (nat, 높을수록 좋다)

| controller | quad_ill_conditioned_d100_k1e+03 | quad_ill_conditioned_d100_k1e+04 | quad_ill_conditioned_d100_k1e+05 | quad_ill_conditioned_d100_k1e+06 |
|---|---|---|---|---|
| `best_static` | 12.741 (n=10) | 8.825 (n=10) | 8.980 (n=10) | 8.724 (n=10) |
| `best_open_loop` | 13.656 (n=10) | 9.332 (n=10) | 9.251 (n=10) | 9.086 (n=10) |
| `heuristic` | 12.740 (n=10) | 8.825 (n=10) | 8.980 (n=10) | 8.755 (n=10) |
| `onestep_narrow` | 17.032 (n=10) | 10.357 (n=10) | 9.879 (n=10) | 9.833 (n=10) |
| `onestep_absolute` | 17.036 (n=10) | 10.571 (n=10) | 9.882 (n=10) | 9.723 (n=10) |
| `committed_Q4_narrow` | 19.417 (n=10) | 11.345 (n=10) | 10.205 (n=10) | 10.123 (n=10) |
| `shrinking_Q4_narrow` | 19.911 (n=10) | 11.222 (n=10) | 10.186 (n=10) | 10.032 (n=10) |

paired delta (nat, 양수면 treatment 가 좋다)

| 비교 | 범위 | median | 95% CI | p | n | 양수 |
|---|---|---|---|---|---|---|
| A2 `shrinking_Q4_narrow` − `best_static` | ALL | +1.690 | [+1.462, +2.368] | <0.0001 | 40 | 40/40 |
| A2 `shrinking_Q4_narrow` − `best_static` | quad_ill_conditioned_d100_k1e+03 | +6.983 | [+6.956, +7.101] | 0.0020 | 10 | 10/10 |
| A2 `shrinking_Q4_narrow` − `best_static` | quad_ill_conditioned_d100_k1e+04 | +2.317 | [+1.529, +2.507] | 0.0020 | 10 | 10/10 |
| A2 `shrinking_Q4_narrow` − `best_static` | quad_ill_conditioned_d100_k1e+05 | +1.399 | [+0.953, +1.517] | 0.0020 | 10 | 10/10 |
| A2 `shrinking_Q4_narrow` − `best_static` | quad_ill_conditioned_d100_k1e+06 | +1.253 | [+0.966, +1.693] | 0.0020 | 10 | 10/10 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | ALL | +0.456 | [+0.254, +0.720] | <0.0001 | 40 | 35/40 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | quad_ill_conditioned_d100_k1e+03 | +2.793 | [+1.462, +3.626] | 0.0020 | 10 | 10/10 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | quad_ill_conditioned_d100_k1e+04 | +0.448 | [+0.238, +1.111] | 0.0371 | 10 | 9/10 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | quad_ill_conditioned_d100_k1e+05 | +0.289 | [+0.009, +0.654] | 0.0488 | 10 | 8/10 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | quad_ill_conditioned_d100_k1e+06 | +0.126 | [-0.041, +0.389] | 0.1309 | 10 | 8/10 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | ALL | +0.010 | [-0.033, +0.053] | 0.9725 | 40 | 21/40 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | quad_ill_conditioned_d100_k1e+03 | +0.472 | [-0.168, +0.693] | 0.1934 | 10 | 7/10 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | quad_ill_conditioned_d100_k1e+04 | -0.019 | [-0.916, +0.078] | 0.3008 | 10 | 4/10 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | quad_ill_conditioned_d100_k1e+05 | +0.009 | [-0.295, +0.077] | 1.0000 | 10 | 5/10 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | quad_ill_conditioned_d100_k1e+06 | +0.000 | [-0.423, +0.069] | 0.7695 | 10 | 5/10 |
| open_loop `best_open_loop` − `best_static` | ALL | +0.395 | [+0.350, +0.476] | <0.0001 | 40 | 40/40 |
| open_loop `best_open_loop` − `best_static` | quad_ill_conditioned_d100_k1e+03 | +1.026 | [+0.862, +1.132] | 0.0020 | 10 | 10/10 |
| open_loop `best_open_loop` − `best_static` | quad_ill_conditioned_d100_k1e+04 | +0.405 | [+0.305, +0.495] | 0.0020 | 10 | 10/10 |
| open_loop `best_open_loop` − `best_static` | quad_ill_conditioned_d100_k1e+05 | +0.307 | [+0.267, +0.389] | 0.0020 | 10 | 10/10 |
| open_loop `best_open_loop` − `best_static` | quad_ill_conditioned_d100_k1e+06 | +0.350 | [+0.267, +0.403] | 0.0020 | 10 | 10/10 |

탐색 비용과 거절률 (전체 인스턴스 median)

| controller | decision-search GE | object GE | 거절률 |
|---|---|---|---|
| `best_static` | 0 | 147.6 | 0.00 |
| `best_open_loop` | 0 | 144.5 | 0.00 |
| `heuristic` | 0 | 147.6 | 0.00 |
| `onestep_narrow` | 1,186 | 142.7 | 0.00 |
| `onestep_absolute` | 10,084 | 140.2 | 0.00 |
| `committed_Q4_narrow` | 69,401 | 144.0 | 0.00 |
| `shrinking_Q4_narrow` | 194,095 | 146.0 | 0.00 |

## nonlinear diagnostic

rosen_d5. 국소최소점 cap + seed 복제로 사용 불가 (D23)

```text
raw        headroom_nonlinear-diagnostic_step_size_fixed_b8_2a09bd45.jsonl
완료 run   72
best_open_loop -> open_loop[1]
best_static -> static[6]
```

절대 median logΔ (nat, 높을수록 좋다)

| controller | rosen_d5_s100_std |
|---|---|
| `best_static` | 1.817 (n=3) |
| `best_open_loop` | 1.817 (n=3) |
| `heuristic` | 1.817 (n=3) |
| `onestep_narrow` | 1.817 (n=3) |
| `onestep_absolute` | 1.817 (n=3) |
| `committed_Q4_narrow` | 1.817 (n=3) |
| `shrinking_Q4_narrow` | 1.817 (n=3) |

paired delta (nat, 양수면 treatment 가 좋다)

| 비교 | 범위 | median | 95% CI | p | n | 양수 |
|---|---|---|---|---|---|---|
| A2 `shrinking_Q4_narrow` − `best_static` | ALL | +0.000 | [+0.000, +0.000] | n/a | 3 | 3/3 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | ALL | -0.000 | [-0.000, -0.000] | n/a | 3 | 0/3 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | ALL | +0.000 | [+0.000, +0.000] | n/a | 3 | 0/3 |
| open_loop `best_open_loop` − `best_static` | ALL | +0.000 | [+0.000, +0.000] | n/a | 3 | 3/3 |

탐색 비용과 거절률 (전체 인스턴스 median)

| controller | decision-search GE | object GE | 거절률 |
|---|---|---|---|
| `best_static` | 0 | 147.2 | 0.19 |
| `best_open_loop` | 0 | 145.4 | 0.32 |
| `heuristic` | 0 | 147.8 | 0.33 |
| `onestep_narrow` | 1,582 | 149.0 | 0.15 |
| `onestep_absolute` | 13,106 | 146.4 | 0.16 |
| `committed_Q4_narrow` | 26,042 | 149.6 | 0.69 |
| `shrinking_Q4_narrow` | 43,011 | 149.6 | 0.69 |

## micro-neural 0bec1125

D27/D28/D29

```text
raw        headroom_micro-neural_step_size_fixed_b8_0bec1125.jsonl
완료 run   216
best_open_loop -> open_loop[0]
best_static -> static[4]
```

절대 median logΔ (nat, 높을수록 좋다)

| controller | mlp_d32_h128_c5_n512_cs128 | mlp_d32_h128_c5_n512_cs64 | mlp_d32_h128_c5_n512_fb |
|---|---|---|---|
| `best_static` | 3.765 (n=3) | 3.029 (n=3) | 5.220 (n=3) |
| `best_open_loop` | 2.100 (n=3) | 2.984 (n=3) | 3.299 (n=3) |
| `heuristic` | 2.162 (n=3) | 0.986 (n=3) | 5.105 (n=3) |
| `onestep_narrow` | 4.296 (n=3) | 3.468 (n=3) | 19.886 (n=3) |
| `onestep_absolute` | 4.642 (n=3) | 3.732 (n=3) | 19.460 (n=3) |
| `committed_Q4_narrow` | -0.402 (n=3) | 1.086 (n=3) | 20.491 (n=3) |
| `shrinking_Q4_narrow` | 3.185 (n=3) | 2.757 (n=3) | 20.395 (n=3) |

paired delta (nat, 양수면 treatment 가 좋다)

| 비교 | 범위 | median | 95% CI | p | n | 양수 |
|---|---|---|---|---|---|---|
| A2 `shrinking_Q4_narrow` − `best_static` | ALL | +0.641 | [-0.900, +15.176] | 0.3594 | 9 | 5/9 |
| A2 `shrinking_Q4_narrow` − `best_static` | mlp_d32_h128_c5_n512_cs128 | -0.900 | [-1.226, +1.052] | n/a | 3 | 1/3 |
| A2 `shrinking_Q4_narrow` − `best_static` | mlp_d32_h128_c5_n512_cs64 | -0.277 | [-0.768, +0.641] | n/a | 3 | 1/3 |
| A2 `shrinking_Q4_narrow` − `best_static` | mlp_d32_h128_c5_n512_fb | +15.176 | [+14.821, +15.512] | n/a | 3 | 3/3 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | ALL | -0.118 | [-1.365, +0.846] | 0.4258 | 9 | 4/9 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | mlp_d32_h128_c5_n512_cs128 | -1.111 | [-1.503, +0.029] | n/a | 3 | 1/3 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | mlp_d32_h128_c5_n512_cs64 | -0.716 | [-1.365, +0.846] | n/a | 3 | 1/3 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | mlp_d32_h128_c5_n512_fb | +0.547 | [-0.118, +0.916] | n/a | 3 | 2/3 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | ALL | +1.666 | [-0.095, +3.168] | 0.0273 | 9 | 7/9 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | mlp_d32_h128_c5_n512_cs128 | +2.941 | [+2.612, +5.210] | n/a | 3 | 3/3 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | mlp_d32_h128_c5_n512_cs64 | +1.666 | [+0.220, +3.168] | n/a | 3 | 3/3 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | mlp_d32_h128_c5_n512_fb | -0.095 | [-0.588, +0.916] | n/a | 3 | 1/3 |
| open_loop `best_open_loop` − `best_static` | ALL | -1.096 | [-1.921, +0.004] | 0.0195 | 9 | 2/9 |
| open_loop `best_open_loop` − `best_static` | mlp_d32_h128_c5_n512_cs128 | -1.522 | [-1.717, -0.670] | n/a | 3 | 0/3 |
| open_loop `best_open_loop` − `best_static` | mlp_d32_h128_c5_n512_cs64 | +0.004 | [-0.697, +0.564] | n/a | 3 | 2/3 |
| open_loop `best_open_loop` − `best_static` | mlp_d32_h128_c5_n512_fb | -1.921 | [-2.362, -1.096] | n/a | 3 | 0/3 |

탐색 비용과 거절률 (전체 인스턴스 median)

| controller | decision-search GE | object GE | 거절률 |
|---|---|---|---|
| `best_static` | 0 | 149.3 | 0.10 |
| `best_open_loop` | 0 | 148.8 | 0.20 |
| `heuristic` | 0 | 141.0 | 0.23 |
| `onestep_narrow` | 1,935 | 148.7 | 0.03 |
| `onestep_absolute` | 11,541 | 148.4 | 0.00 |
| `committed_Q4_narrow` | 21,449 | 148.8 | 0.66 |
| `shrinking_Q4_narrow` | 265,056 | 146.9 | 0.00 |

## micro-neural 7aac1b26

D27/D28/D29

```text
raw        headroom_micro-neural_step_size_fixed_b8_7aac1b26.jsonl
완료 run   144
best_open_loop -> open_loop[0]
best_static -> static[4]
```

절대 median logΔ (nat, 높을수록 좋다)

| controller | mlp_d32_h128_c5_n512_cs64 | mlp_d32_h128_c5_n512_fb |
|---|---|---|
| `best_static` | 3.029 (n=3) | 5.220 (n=3) |
| `best_open_loop` | 2.984 (n=3) | 3.299 (n=3) |
| `heuristic` | 0.986 (n=3) | 5.105 (n=3) |
| `onestep_narrow` | 3.468 (n=3) | 19.886 (n=3) |
| `onestep_absolute` | 3.732 (n=3) | 19.460 (n=3) |
| `committed_Q4_narrow` | 1.086 (n=3) | 20.491 (n=3) |
| `shrinking_Q4_narrow` | 2.757 (n=3) | 20.395 (n=3) |

paired delta (nat, 양수면 treatment 가 좋다)

| 비교 | 범위 | median | 95% CI | p | n | 양수 |
|---|---|---|---|---|---|---|
| A2 `shrinking_Q4_narrow` − `best_static` | ALL | +7.731 | [-0.523, +15.344] | 0.2188 | 6 | 4/6 |
| A2 `shrinking_Q4_narrow` − `best_static` | mlp_d32_h128_c5_n512_cs64 | -0.277 | [-0.768, +0.641] | n/a | 3 | 1/3 |
| A2 `shrinking_Q4_narrow` − `best_static` | mlp_d32_h128_c5_n512_fb | +15.176 | [+14.821, +15.512] | n/a | 3 | 3/3 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | ALL | +0.214 | [-1.041, +0.881] | 1.0000 | 6 | 3/6 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | mlp_d32_h128_c5_n512_cs64 | -0.716 | [-1.365, +0.846] | n/a | 3 | 1/3 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | mlp_d32_h128_c5_n512_fb | +0.547 | [-0.118, +0.916] | n/a | 3 | 2/3 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | ALL | +0.568 | [-0.342, +2.417] | 0.2188 | 6 | 4/6 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | mlp_d32_h128_c5_n512_cs64 | +1.666 | [+0.220, +3.168] | n/a | 3 | 3/3 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | mlp_d32_h128_c5_n512_fb | -0.095 | [-0.588, +0.916] | n/a | 3 | 1/3 |
| open_loop `best_open_loop` − `best_static` | ALL | -0.897 | [-2.142, +0.284] | 0.1562 | 6 | 2/6 |
| open_loop `best_open_loop` − `best_static` | mlp_d32_h128_c5_n512_cs64 | +0.004 | [-0.697, +0.564] | n/a | 3 | 2/3 |
| open_loop `best_open_loop` − `best_static` | mlp_d32_h128_c5_n512_fb | -1.921 | [-2.362, -1.096] | n/a | 3 | 0/3 |

탐색 비용과 거절률 (전체 인스턴스 median)

| controller | decision-search GE | object GE | 거절률 |
|---|---|---|---|
| `best_static` | 0 | 148.9 | 0.10 |
| `best_open_loop` | 0 | 148.4 | 0.17 |
| `heuristic` | 0 | 139.5 | 0.20 |
| `onestep_narrow` | 2,174 | 148.8 | 0.02 |
| `onestep_absolute` | 10,881 | 148.4 | 0.00 |
| `committed_Q4_narrow` | 23,917 | 148.6 | 0.38 |
| `shrinking_Q4_narrow` | 211,862 | 148.2 | 0.00 |

## micro-neural 9f3194be

D27/D28/D29

```text
raw        headroom_micro-neural_step_size_fixed_b8_9f3194be.jsonl
완료 run   216
best_open_loop -> open_loop[1]
best_static -> static[4]
```

절대 median logΔ (nat, 높을수록 좋다)

| controller | mlp_d32_h128_c5_n512_cs128 | mlp_d32_h128_c5_n512_cs64 | mlp_d32_h128_c5_n512_fb |
|---|---|---|---|
| `best_static` | 2.219 (n=3) | 1.120 (n=3) | 5.109 (n=3) |
| `best_open_loop` | 1.635 (n=3) | 1.013 (n=3) | 3.482 (n=3) |
| `heuristic` | 0.580 (n=3) | 0.455 (n=3) | 5.105 (n=3) |
| `onestep_narrow` | 1.362 (n=3) | 0.575 (n=3) | 19.056 (n=3) |
| `onestep_absolute` | 1.809 (n=3) | 1.241 (n=3) | 19.310 (n=3) |
| `committed_Q4_narrow` | 0.170 (n=3) | 0.073 (n=3) | 19.527 (n=3) |
| `shrinking_Q4_narrow` | 1.657 (n=3) | 1.002 (n=3) | 19.299 (n=3) |

paired delta (nat, 양수면 treatment 가 좋다)

| 비교 | 범위 | median | 95% CI | p | n | 양수 |
|---|---|---|---|---|---|---|
| A2 `shrinking_Q4_narrow` − `best_static` | ALL | -0.093 | [-0.598, +14.065] | 0.9102 | 9 | 3/9 |
| A2 `shrinking_Q4_narrow` − `best_static` | mlp_d32_h128_c5_n512_cs128 | -0.598 | [-0.626, -0.392] | n/a | 3 | 0/3 |
| A2 `shrinking_Q4_narrow` − `best_static` | mlp_d32_h128_c5_n512_cs64 | -0.093 | [-0.442, -0.086] | n/a | 3 | 0/3 |
| A2 `shrinking_Q4_narrow` − `best_static` | mlp_d32_h128_c5_n512_fb | +14.065 | [+13.470, +14.250] | n/a | 3 | 3/3 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | ALL | -0.080 | [-0.118, +0.918] | 0.6523 | 9 | 4/9 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | mlp_d32_h128_c5_n512_cs128 | -0.118 | [-0.123, +1.128] | n/a | 3 | 1/3 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | mlp_d32_h128_c5_n512_cs64 | +0.104 | [-0.097, +0.918] | n/a | 3 | 2/3 |
| C2 `shrinking_Q4_narrow` − `onestep_narrow` | mlp_d32_h128_c5_n512_fb | -0.080 | [-0.118, +0.304] | n/a | 3 | 1/3 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | ALL | +0.606 | [-0.949, +1.128] | 0.2031 | 9 | 6/9 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | mlp_d32_h128_c5_n512_cs128 | +1.110 | [+0.426, +1.657] | n/a | 3 | 3/3 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | mlp_d32_h128_c5_n512_cs64 | +0.973 | [+0.606, +1.128] | n/a | 3 | 3/3 |
| C3 `shrinking_Q4_narrow` − `committed_Q4_narrow` | mlp_d32_h128_c5_n512_fb | -0.949 | [-0.989, -0.028] | n/a | 3 | 0/3 |
| open_loop `best_open_loop` − `best_static` | ALL | -0.583 | [-1.585, -0.125] | 0.0039 | 9 | 0/9 |
| open_loop `best_open_loop` − `best_static` | mlp_d32_h128_c5_n512_cs128 | -0.583 | [-0.597, -0.308] | n/a | 3 | 0/3 |
| open_loop `best_open_loop` − `best_static` | mlp_d32_h128_c5_n512_cs64 | -0.125 | [-0.132, -0.075] | n/a | 3 | 0/3 |
| open_loop `best_open_loop` − `best_static` | mlp_d32_h128_c5_n512_fb | -1.585 | [-1.816, -0.665] | n/a | 3 | 0/3 |

탐색 비용과 거절률 (전체 인스턴스 median)

| controller | decision-search GE | object GE | 거절률 |
|---|---|---|---|
| `best_static` | 0 | 146.8 | 0.20 |
| `best_open_loop` | 0 | 144.2 | 0.24 |
| `heuristic` | 0 | 145.6 | 0.12 |
| `onestep_narrow` | 1,206 | 145.5 | 0.35 |
| `onestep_absolute` | 11,030 | 147.6 | 0.00 |
| `committed_Q4_narrow` | 2,822 | 145.0 | 0.89 |
| `shrinking_Q4_narrow` | 31,438 | 146.8 | 0.29 |
