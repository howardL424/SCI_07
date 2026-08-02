# Plan 1 改动过程总结与后续验证待办

> **日期**：2026-08-01  
> **范围**：纯 TD3 稀疏 FF-RLS 参数辨识（Plan 1）  
> **用途**：供后续对话快速恢复上下文；**勿当作已验证通过的结论**  
> **用法文档**：[`TD3_BASELINE_USAGE.md`](TD3_BASELINE_USAGE.md) §7  
> **Plan 文件**（勿在实现中编辑）：  
> - Plan 1：`C:\Users\LTY\.cursor\plans\参数辨识接入方案_23b78aa2.plan.md`  
> - Plan 2（未启动）：`C:\Users\LTY\.cursor\plans\专家轨迹与BC参数辨识扩展.plan.md`

---

## 1. 背景与目标

论文扩展版测量预处理采用**三状态 Kalman**（`r,ṙ,r̈`），回归用**因果 FF-RLS + L1 soft-threshold**，估计制导式(47) 的 `Q/b/v_r`，以降低带噪解析系数抖动。

用户约束（对话中多次确认）：

| 约束 | 落实 |
|------|------|
| 测量预处理用 Kalman（扩展 HTML §4.2），不是低通 | 已实现 |
| 首轮正式设置 `rls_mu>0`，保留 `mu=0` 开关 | 默认 `1e-3`（曾误改为 0，已回退；CLI 可关） |
| **不**给式(12) 注入过程噪声 | `kf_qj` 仅 KF 模型参数 |
| 首版不做 PE / dead-zone / trace-limit / 自适应 λ | 未实现 |
| Policy 仍看干净 13D；噪声只走系数路径 | 已实现 |
| Plan 1 **只**接纯 TD3；不改专家库 / BC-TD3 | 已遵守 |
| Plan 2 须等 Plan 1 验证通过后再做 | **未启动** |

临时测量噪声（可改 CLI，尚未最终定稿）：

- `σ_q = 0.1°`，`σ_q̇ = 0.001 rad/s`
- 姿态暂与 `σ_q` 同量级
- **`σ_r` / `σ_ṙ`（2026-08-01 调整）**：默认由占位 `10 m / 1 m/s` 改为 **`0 / 0`**（本轮影子诊断先去掉距离通道噪声；原占位值保留备查，5.2 标定后再定稿）
- 须写入运行配置，不得静默硬编码后遗忘

---

## 2. 本次实际改动清单

### 2.1 制导入口拆分

- 文件：`evasion_helpers.m`
- 新增 `computeOptimalCmd_duo_target_coeffs`（直接注入 `Q,b,v_r,r,ṙ,tgo`）
- 原 `computeOptimalCmd_duo_target` 解析后委托 coeffs，保持 `analytic_truth` 等价

### 2.2 辨识模块（新建）

| 文件 | 职责 |
|------|------|
| `rl_td3/matlab/stage2_ident_init.m` | 三通道 D1/D2/T：KF 状态 + RLS `θ/P` + 噪声/超参 |
| `rl_td3/matlab/stage2_ident_step.m` | 加噪 → KF → 因果 RLS → L1 → 输出系数与诊断 |
| `rl_td3/matlab/stage2_ident_commit.m` | 提交本拍饱和 `a_M`，供下一拍回归 |

**因果约定**（实现时必须保持）：

1. 本拍 KF 得到 `r̈̂(k)`  
2. RLS：`y=r̈̂(k)`，`φ=[r̂(k-1); a_M_sat(k-1); 1]`（T 通道无常数项维数不同）  
3. 更新后的 `θ̂(k)` 才生成本拍指令  
4. 首拍只 warm-start，**不做** RLS；绝不使用 `state(k+1)` 反推本拍命令  

注意：输出诊断结构体命名为 `id_diag`，**禁止**用变量名 `diag`（会遮蔽 MATLAB 内置 `diag`）。

### 2.3 环境接入

| 文件 | 改动 |
|------|------|
| `rl_td3/matlab/stage2_core_init.m` | 默认 `coeff_mode=analytic_truth` + 噪声/KF/RLS 字段 |
| `rl_td3/matlab/stage2_step.m` | 可选 `ident_state`；模式选择；Q·tgo 钳制与解析路径一致 |
| `rl_td3/matlab/env_step.m` | persistent `ident_state`；`t≈0` 与 cfg cache key 变化时重置 |

`coeff_mode`：

- `analytic_truth`：旧路径（默认）
- `analytic_noisy`：带噪测量 → 解析 `computeBC`
- `sparse_ffrls`：KF + L1 FF-RLS；`ident_apply_cmd=false` 为影子模式

防御弹 PN + RK4 植物仍用**真值几何**。

### 2.4 纯 TD3 Python 入口

| 文件 | 改动 |
|------|------|
| `rl_td3/ident_cfg.py` | **新建**：CLI 参数、`matlab_cfg_from_args`、`save_ident_run_config` |
| `td3_baseline/train_td3_pure.py` | 注入辨识 CLI；train/eval env 共用 `matlab_cfg`；写 `ident_run_config.json` |
| `rl_td3/evaluate_td3.py` | 同名只读评估参数（`evaluate_td3_pure.py` 转发即可用） |
| `td3_baseline/run_ident_ablation_pure.py` | **新建**：同 IC / 同噪声配对四模式消融 |
| `td3_baseline/md document/TD3_BASELINE_USAGE.md` | 增补 §7 用法 |

**未改**（Plan 1 明确禁止）：`expert_trajectory/`、`rl_td3/train_td3.py`、`td3_bc.py`、BC 数据加载器。

### 2.5 测试

- `rl_td3/matlab/test_stage2_ident.m`：回归 / 无噪 KF / 合成 RLS / 因果 / 重置 / coeffs 等价  
- 2026-08-01 本地 MATLAB R2024a：`[PASS] test_stage2_ident 全部通过`（含 σ 默认改为 0 后复跑）

### 2.6 影子诊断落盘（2026-08-01）

| 文件 | 改动 |
|------|------|
| `stage2_ident_step.m` | `id_diag` 增补 `r_true/rdot_true/rddot_true`（`r̈` 用真值 `ṙ` 有限差分） |
| `stage2_step.m` | 估参指令 vs 采用指令 → `a_cmd_err` |
| `env_step.m` | info_vec 扩展至 32 维（旧脚本只读前 9 项仍兼容） |
| `evasion_env.py` | 解析为 `info["ident_diag"]` |
| `run_ident_shadow_diag_pure.py` | **新建**：影子逐步 RMSE / 收敛步 / 落盘 |

默认 `σ_r/σ_ṙ`：`10/1` → `0/0`（见 §1、§4）。  
默认 `rls_mu`：曾误改为 `0`（依据已作废的 5.2），**已回退为 `1e-3`**。

### 2.7 超参网格扫描（2026-08-01）

| 文件 | 改动 |
|------|------|
| `run_ident_hyperparam_scan_pure.py` | **新建**：影子模式下 `kf_qj×λ×μ` 扫描 + μ vs μ=0 汇总 |

---

## 3. 首轮配对消融结果（仅供参考，样本小）

- Checkpoint：`td3_baseline/checkpoints/training_300k_20260730_1917/best/best_model`
- 输出目录：`.../ident_ablation_plan1/`（`summary.json` + `metrics_*.npz`）
- 协议：前 5 个 test IC；每局 `noise_seed + local_idx`；含 `--also-mu0`

| 模式 | both_ok | 平均 r_MD1 / r_MD2 | 备注 |
|------|---------|-------------------|------|
| analytic_truth | 100% | 37.6 / 34.9 | 干净基准 |
| analytic_noisy | 80% | 35.0 / 31.3 | IC4 拦截 |
| sparse_ffrls (μ=1e-3) | 100% | 314 / 116 | 脱靶明显偏大（更保守） |
| sparse_ffrls (μ=0) | 100% | 370 / 185 | 与 μ>0 接近；IC3 差异大 |

**初步解读（非正式结论）**：相对 `analytic_noisy` 恢复了 both_ok，但 miss 距离系统性偏大，不能单凭 both_ok 宣称辨识成功；需影子 RMSE + 更大样本 + 噪声标定后再定是否重训。

---

## 4. 当前默认值速查

```text
coeff_mode       = analytic_truth
ident_apply_cmd  = true（--ident-shadow 关闭指令注入）
sigma_q          = 0.1 * pi/180
sigma_qdot       = 0.001
sigma_theta/psi  = 同 sigma_q
sigma_r / rdot   = 0 / 0   ← ⚠ 原占位 10/1；σ=0 使 KF 退化，待重定（§5.2.2）
kf_qj            = 1e4
rls_lambda       = 0.98
rls_mu           = 1e-3    ← 曾误改为 0，已回退
rls_P0           = 1e4
noise_seed       = 0
```

⚠ `σ_r/σ_ṙ=0` **不是经过验证的推荐值**（使 KF 退化，见 §5.2.2）；`rls_mu=1e-3` 已从误改回退，μ 净贡献仍待有效配置下重判。

历史占位（消融 `ident_ablation_plan1` 用）：`σ_r=10`、`σ_ṙ=1`、`rls_mu=1e-3`。
Python：`E:\Python\venvs\rl_pytorch\Scripts\python.exe`  
工作目录：`D:\LTY\yan\yanerxia\GAIL`

---

## 5. 后续验证待办（给其它对话用）

> 状态约定：`[ ]` 未做 / `[~]` 进行中 / `[x]` 完成。完成一项时请在本文件勾选并补一行结果摘要与日期。

### 5.1 影子诊断（优先）

- [x] 固定纯 TD3 checkpoint，跑 `--coeff-mode sparse_ffrls --ident-shadow`
- [x] 记录相对真值：KF `r/ṙ/r̈` 误差、11 维 θ RMSE、收敛步数、指令预测误差
- [x] 若影子参数已很差：先扫 `σ_r/σ_ṙ`、`kf_qj`、`rls_lambda/mu`，**不要**急着闭环下结论  
  → **本轮结论：影子 θ / 指令已很差，进入 5.2 标定前不要据此下闭环结论**
- [x] （可选）扩展 `stage2_ident_step` / 消融脚本，把 `ident_diag` 逐步落盘，避免只看闭环 miss  
  → 新增 `td3_baseline/run_ident_shadow_diag_pure.py`；`env_step` info_vec 扩展至 32 维

**结果摘要（2026-08-01）**

- Checkpoint：`training_300k_20260730_1917/best/best_model`
- 输出：`.../ident_shadow_diag_plan1/`（`summary.json` + `shadow_diag_series.npz` + `ident_run_config.json`）
- 协议：前 5 个 test IC；`σ_r=σ_ṙ=0`（本轮新默认）；`rls_mu=1e-3`；`noise_seed+i`
- 末段均值（后 20%）：

| 指标 | 均值 |
|------|------|
| KF `r` RMSE | ~3e-15（数值零） |
| KF `ṙ` RMSE | ~5e-15（数值零） |
| KF `r̈` RMSE | ~486 |
| 11 维 θ RMSE | ~2e6（IC3 爆到 ~1e7） |
| 指令预测 RMSE | ~103 m/s² |
| 收敛率（θ RMSE&lt;0.5 且末段达标） | **0/5** |

解读：距离通道无噪时 KF 的 `r/ṙ` 跟踪几乎完美，但 `r̈` 与 θ 严重失真，估参指令相对真值偏差 ~100 m/s² 量级。问题不在 `σ_r/σ_ṙ` 占位。

> **后续修正（§5.2.1）**：θ RMSE 是把 A21(~1e-4)、B(~1)、C1(~10) 混在一起算的，数值被 C1 主导，对最敏感的 `Q=√A21` 几乎不反映；`a_cmd_err` 用的是**未饱和**指令，而上限只有 8g≈78.5 m/s²，所以 ~100 的数字偏大。这两个指标只适合做相对比较，不宜按绝对值解读。

### 5.2 噪声与超参标定 —— ⚠ 结论已作废，需重做

- [x] 确认或替换占位 `σ_r`、`σ_ṙ`（及文档/运行 JSON 同步）  
  → 临时改 `0/0`（原占位 `10/1`）。**但见 §5.2.2：`σ=0` 本身让 KF 退化，这个取值需要重新考虑**
- [~] 小网格扫描：`kf_qj` × `rls_lambda` × `rls_mu∈{0, 1e-3, …}`  
  → 脚本 `run_ident_hyperparam_scan_pure.py`；输出 `.../ident_hyperparam_scan_plan1/`  
  → **扫描已跑完但结果无效**（两个旋钮在该配置下不起作用，见 §5.2.2）
- [ ] 判断 μ>0 是否相对 μ=0 有净贡献  
  → ~~无净贡献~~ **该判断不成立**：测试量级下 L1 几乎不作用，等于没测

**原始扫描数据（2026-08-01，仅存档，勿引用其结论）**

- 协议：同 5.1 影子；5 IC；`σ_r=σ_ṙ=0`
- 网格：`kf_qj∈{1e2,1e3,1e4,1e5}` × `λ∈{0.95,0.98,0.995}` × `μ∈{0,1e-4,1e-3,1e-2}`（48 点）

| 配置 | med a_cmd | med θ | 备注 |
|------|-----------|-------|------|
| `qj=1e4,λ=0.98,μ=1e-3` | 98.7 | 548 | 原默认 |
| `qj=1e4,λ=0.98,μ=0` | 98.7 | 548 | 与 μ>0 相同 |
| `qj=1e2,λ=0.95,μ=0.01` | 94.4 | 555 | 表面最优 a_cmd |
| `qj=1e5,λ=0.995,μ=0` | 104 | 180 | 表面最优 θ；实为浮点伪影 |

**为什么无效**：全部 48 个配置的 `med_rddot` 都等于 494（完全不随 `qj` 变），这是旋钮失效的直接信号。原因见 §5.2.2。

---

### 5.2.1 根因诊断（2026-08-01，三个针对性实验）

复现脚本：`rl_td3/matlab/tmp_ident_probe.m`（临时诊断，未纳入测试套件）。

**① 回归方程与因果时序是对的，不是元凶**

用真值 θ 反推 `r̈`，对比不同配对（基准 `std(r̈_true)=7.66`）：

| 配对 | RMSE | corr |
|------|------|------|
| `φ(k)·θ(k)` 同拍 | 5.06 | 0.779 |
| **`φ(k-1)·θ(k)`（当前实现）** | **0.69** | **1.000** |
| `φ(k-1)·θ(k-1)` 全滞后 | 0.24 | 1.000 |

当前因果约定几乎精确。KF 在平滑段 `r̈` RMSE=0.85（vs 信号 std 7.66）也够用。  
（次要可改进项：真值关系其实是 `y(k)=φ(k-1)·θ(k-1)`，即 RLS 实际估到的是 `θ(k-1)` 却当作 `θ(k)` 用；θ 每步变 2–10%，属小量额外滞后。）

**② `σ_r=σ_ṙ=0` 使 KF 退化，`kf_qj` 与 `rls_mu` 变成"死旋钮"**

- `Rxi = diag([max(σ_r²,1e-18), max(σ_ṙ²,1e-18)])`：σ=0 时 `R≈0`，卡尔曼增益 `K = P⁻H'(HP⁻H')⁻¹` **对 P 的整体缩放不变** → `kf_qj` 完全不影响稳态增益。数值验证：`qj` 从 `1e2` 到 `1e5`，稳态 `K₃=[-585.78643763, 37.57359313]` 到 7 位有效数字完全相同。
- 这直接解释了 48 个配置 `med_rddot≡494`。
- `qj=1e5` 处 `θ` 从 548 掉到 180：此时 `R/Q≈3e-20`，已达双精度极限，是**浮点伪影**，不是真实改善。
- `μ≤1e-2` 对量级 O(1)~O(1e3) 的 `θ_sc` 做软阈值，作用可忽略 → μ 的对照等于没做。

**③ 真正的根因：参数强时变 + 一个方向缺乏激励**

| 现象 | 实测（D1 通道，单局 60 步） |
|------|------|
| C1 波动 | std/\|mean\| = 1.80，局内摆动 22.9× |
| B2 波动 | 局内摆动 1100× |
| A21 波动 | 局内摆动 11.7× |
| 相邻步 \|Δθ\|/\|θ\| 中位数 | A21 9.7% / B1 1.7% / B2 9.7% / C1 4.5% |
| Gram 特征值（缩放后） | [8.2e-5, 0.56, 0.67, 2.37] |
| 定常 θ 批量 LS | R²=0.83，残差 3.16 **vs** 真值时变 θ 的 0.69 |

- Gram 最小特征值比最大低 **4 个量级** → 有一个参数方向基本没被激励；配合 `P0=1e4` 且无 PE 门控 / dead-zone，协方差在该方向发散，θ 随之漂走。
- `λ=0.98` 记忆窗约 50 步，而 θ 在同一窗内变化 10–1000×：**FF-RLS 的"参数准定常"前提不成立**。
- 即使取整窗最优的**常数** θ，残差也比真值时变 θ 差 **4.6 倍** —— 这是任何窗口估计器的性能上限。

结论：`λ/μ/qj` 怎么调都救不回来，**问题在辨识目标的设定，不在超参**。

**④ 顺带发现：`analytic_noisy` 基线存在真值泄漏**

`stage2_ident_step.m` 计算 `analytic_noisy` 的 `C1` 时直接传入真值 `a_D1/a_D2`（未加噪、未估计）：

```matlab
[B11n, B21n, C11n] = computeBC(mD1.qy, mD1.qz, theta_M_m, psi_M_m, ...
    theta_D1_m, psi_D1_m, a_D1(1), a_D1(2));   % a_D 为真值
```

而 `a_D`（防御弹加速度）恰是突防弹**唯一真正测不到**的量。因此 `analytic_noisy` 基线偏乐观，§3 的配对消融对比也受此影响。

---

### 5.2.2 建议方向（待决策，尚未实施）

A21、B1、B2 都是纯几何/姿态量，可由 `q, q̇, θ_M, ψ_M` 直接解析算出（本来就可测）；只有 C1 依赖未知的 `a_D`。

建议把 11 维回归缩成**每通道只估 C1（即 `v_r`）**：激励充分（回归量退化为常数项 + 残差滤波）、问题良态，且正好对准真实未知量。同时修掉 ④ 的 `a_D` 泄漏，让基线可信。

**在此之前需要修正的遗留项**

- [x] `rls_mu` 默认曾误改为 `0` → **已回退为 `1e-3`**；μ 净贡献仍须在有效配置下重判
- [ ] `σ_r/σ_ṙ = 0` 需重新考虑：要么恢复非零，要么把 KF 的 `R` 与真实 σ **解耦**、作为独立设计参数（当前二者绑定，是 ② 的直接成因）；**按用户要求暂保持 0**
- [ ] `rls_P0=1e4` 在缺激励方向会发散，需配 trace-limit 或改设计

### 5.3 扩大闭环配对消融

> **暂缓**：§5.2.1 已证明当前 `sparse_ffrls` 的辨识目标设定有结构性问题（参数强时变 + 缺激励），
> 且 `analytic_noisy` 基线含 `a_D` 真值泄漏。在这两点修正前扩大样本量只会更精确地测量一个
> 已知失效的配置，不能作为“通过/不通过”的依据。

- [ ] 同一 checkpoint / 同一 test IC 集 / 同一 `noise_seed` 协议
- [ ] 模式：`analytic_truth` | `analytic_noisy` | `sparse_ffrls(μ)` | `sparse_ffrls(μ=0)`
- [ ] 样本量：至少 30–50 IC（理想全量 test）
- [ ] 指标：both_ok、拦截率、`r_min1/2/T`、加速度总变差、饱和率、能量
- [ ] **通过门槛（Plan 1）**：`sparse_ffrls` 相对 `analytic_noisy` 在参数 RMSE **或** 闭环指标上稳定改善  
      （仅 both_ok↑ 但 miss 系统性恶化 → 视为未通过，继续调参）

### 5.4 通过后才做

- [ ] 用 `--coeff-mode sparse_ffrls` **重训**新的纯 TD3（独立 `save-dir`）
- [ ] 新模型 vs 旧 `analytic_truth` 训出模型：同一 MC 协议对比
- [ ] 更新本文件与 `TD3_BASELINE_USAGE.md` 中的推荐默认超参

### 5.5 明确暂缓（未通过前禁止）

- [ ] ~~启动 Plan 2~~：专家轨迹重放 / sidecar / BC-TD3 接入  
- [ ] ~~给式(12) 加过程噪声~~  
- [ ] ~~首版加 PE / dead-zone / trace-limit / 自适应遗忘~~（除非基础 KF+RLS 暴露明确问题）
- [ ] ~~改 13D obs 把噪声喂给 Actor~~（Plan 1 范围外）

---

## 6. 常用命令速查

```powershell
cd D:\LTY\yan\yanerxia\GAIL
$PY = "E:\Python\venvs\rl_pytorch\Scripts\python.exe"
$MODEL = "td3_baseline/checkpoints/training_300k_20260730_1917/best/best_model"

# 单元测试
& "D:\Program Files\MATLAB\R2024a\bin\matlab.exe" -batch `
  "addpath('rl_td3/matlab'); addpath('.'); addpath('expert_trajectory'); ok=test_stage2_ident(); exit(~ok);"

# 配对消融
& $PY td3_baseline/run_ident_ablation_pure.py `
  --model $MODEL --episodes 30 --noise-seed 0 --also-mu0 `
  --out-dir td3_baseline/checkpoints/training_300k_20260730_1917/ident_ablation_N30

# 超参网格（影子）
& $PY td3_baseline/run_ident_hyperparam_scan_pure.py `
  --model $MODEL --episodes 5 --noise-seed 0 `
  --out-dir td3_baseline/checkpoints/training_300k_20260730_1917/ident_hyperparam_scan_plan1

# 影子评估（逐步诊断落盘）
& $PY td3_baseline/run_ident_shadow_diag_pure.py `
  --model $MODEL --episodes 5 --noise-seed 0 `
  --out-dir td3_baseline/checkpoints/training_300k_20260730_1917/ident_shadow_diag_plan1

# 影子闭环对照（评估入口；仅看 miss，不含逐步 RMSE）
& $PY td3_baseline/evaluate_td3_pure.py `
  --model $MODEL --coeff-mode sparse_ffrls --ident-shadow --noise-seed 0

# 闭环评估对照
& $PY td3_baseline/evaluate_td3_pure.py --model $MODEL --coeff-mode analytic_noisy --noise-seed 0
& $PY td3_baseline/evaluate_td3_pure.py --model $MODEL --coeff-mode sparse_ffrls --rls-mu 1e-3 --noise-seed 0
```

---

## 7. 给后续 Agent 的工作约定

1. **先读本文件 + `TD3_BASELINE_USAGE.md` §7**，再改代码。  
2. 默认保持 `analytic_truth`，避免破坏 BC / 历史 MC。  
3. 验证未通过前：**不重训**、**不启动 Plan 2**、**不擅自加 PE 等机制**。  
4. 改辨识逻辑时同步更新本文件「改动清单」与待办勾选。  
5. Python 解释器固定：`E:\Python\venvs\rl_pytorch\Scripts\python.exe`。
