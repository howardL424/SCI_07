# E1V 50k 诊断（早期动态 + BC 对照）

## seed 0：0~25k 训练动态（actor_loss = -Q，因 λ=0）

| total_timesteps | actor_loss(-Q) | critic_loss | ep_rew_mean |
|---|---|---|---|
| 1701 | -2.99 | 377.0 | 109.0 |
| 5098 | -13.2 | 192.0 | 92.5 |
| 9070 | -19.3 | 270.0 | 93.7 |
| 12845 | -29.1 | 21.7 | 96.3 |
| 16293 | -31.1 | 11.5 | 100.0 |
| 19700 | -36.1 | 2.64 | 95.0 |
| 23133 | -35.4 | 9.18 | 82.5 |
| 25815 | -37.7 | 4.68 | 84.5 |

- 早期 critic_loss：max=386.0，末段均值(后1/3)=87.01499999999999
- 早期 actor_loss(-Q)：首=-2.99，末=-37.7

## seed 0：BC 最优 checkpoint 对照（epoch 21）

- val MSE=0.31676385559698456；val MAE=0.41951886119369924；分维 MAE={'rhoE_n': 0.41948717795602336, 'rhoT_n': 0.19422955499796607, 'gamma_n': 0.6448398506271082}
- 训练集分维 MAE={'rhoE_n': 0.35847066795473975, 'rhoT_n': 0.1816187946578231, 'gamma_n': 0.5715403808362944}

## seed 42：0~25k 训练动态（actor_loss = -Q，因 λ=0）

| total_timesteps | actor_loss(-Q) | critic_loss | ep_rew_mean |
|---|---|---|---|
| 1726 | -2.56 | 160.0 | 84.3 |
| 4406 | -12.7 | 25.6 | 90.5 |
| 7768 | -17.5 | 49.6 | 93.2 |
| 11678 | -23.1 | 12.0 | 93.5 |
| 15072 | -28.0 | 23.6 | 95.2 |
| 18525 | -29.3 | 406.0 | 82.0 |
| 22518 | -34.9 | 269.0 | 90.4 |
| 25155 | -39.1 | 9.74 | 97.1 |

- 早期 critic_loss：max=406.0，末段均值(后1/3)=110.37100000000001
- 早期 actor_loss(-Q)：首=-2.56，末=-39.1

## seed 42：BC 最优 checkpoint 对照（epoch 10）

- val MSE=0.29854187155746353；val MAE=0.4113644655383191；分维 MAE={'rhoE_n': 0.379636458546597, 'rhoT_n': 0.18847603242375124, 'gamma_n': 0.6659809056446091}
- 训练集分维 MAE={'rhoE_n': 0.3892488394411678, 'rhoT_n': 0.1863796395347977, 'gamma_n': 0.6067492009233925}

## seed 0 evaluations.npz：keys=['timesteps', 'results', 'ep_lengths', 'successes']
- timesteps: shape=(3,), dtype=int64
- results: shape=(3, 150), dtype=float64
- ep_lengths: shape=(3, 150), dtype=int64
- successes: shape=(3, 150), dtype=bool

## seed 42 evaluations.npz：keys=['timesteps', 'results', 'ep_lengths', 'successes']
- timesteps: shape=(3,), dtype=int64
- results: shape=(3, 150), dtype=float64
- ep_lengths: shape=(3, 150), dtype=int64
- successes: shape=(3, 150), dtype=bool

