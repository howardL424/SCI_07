# 上下文纪要：专家轨迹批量生成（GAIL 预训练数据）

> 本文件供新对话直接续接工作。前序纪要见 `CONTEXT_SUMMARY_伪谱专家轨迹.md`（单打靶求解器建立阶段）。  
> 任务：以单打靶求解器（`solve_ss_trajectory`）为阶段B，批量生成 1000 条专家轨迹，输出 `expert_lib_1000.mat`，供 GAIL 预训练。

---

## 0. 核心设定（已确认，勿改）

- **决策变量**：θ(t) = [γ, wT, wE]，K=10 段分段常值；w 固定 = 1e4；d* 固定 = 20m。
- **GAIL state（13维相对态势量）**：`[r_MD1, rdot_MD1, qy_D1, qz_D1, r_MD2, rdot_MD2, qy_D2, qz_D2, r_MT, qy_MT, qz_MT, V_M, tgo_norm]`。使用相对量而非绝对 18 维，原因：3D + 大范围随机初态，绝对坐标分布发散，网络难以泛化。
- **GAIL action（4维）**：`[w=1e4, γ(t), wT(t), wE(t)]`，分段常值，按 K 段取。
- **expert_sa 格式**：`[N_total_steps × 17]`，第 1-13 列 = state，第 14-17 列 = action（w/γ/wT/wE）。
- **`samples_clean.mat` 14 列**仅用于初始化几何（重建 M0/D10/D20），不进 GAIL state，进 meta 备查。
- **采样密度**：dt=0.1s，每条约 80-150 步，1000 条约 8-15 万对。

---

## 1. 已完成的代码修改与功能

### 1.1 文件清单与状态

| 文件 | 本轮改动 | 状态 |
|---|---|---|
| `solve_ss_trajectory.m` | `max_iter` 默认从 500 降到 **300**（批处理失败条快速退出）；新增 `res.theta_K`（3×K 原始分段权重，供 `extract` 用）；`ok=false` 时同样从 `opti.debug.value` 提取 `theta_K` | ✅ |
| `extract_expert_transitions.m` | **新建**：数值 RK4 重跑规避段（dt=0.1s），每步提取 13D 相对态势 state + 4D action + 18D 绝对态（备查）；theta 按当前 K 段分段常值取 | ✅ |
| `generate_ps_expert_library.m` | **重写**：接入阶段B（`solve_ss_trajectory`）+ ok/debug 降级逻辑 + 调 `extract_expert_transitions` + 分段 checkpoint 保存（每 `checkpoint_every` 条）+ 最终保存 `expert_lib.mat` | ✅ |

### 1.2 `generate_ps_expert_library` 主要 opts 字段

```matlab
opts.run_stageB       = true;        % 开启阶段B NLP
opts.plot             = false;       % 批处理时关图
opts.save             = true;        % 开启保存
opts.save_file        = 'expert_lib_1000.mat';
opts.checkpoint_every = 50;          % 每 50 条保存一次 *_ckpt0050.mat 等
opts.ss_cfg           = struct();    % 传给 solve_ss_trajectory (可设 max_iter/nsub 等)
opts.ex_cfg           = struct();    % 传给 extract_expert_transitions (可设 dt_out 等)
```

### 1.3 降级逻辑（自动，无需手动操作）

```
IPOPT 求解 → ok=true  → 直接入库（标记 ok_mode='ok'）
           → ok=false → 取 opti.debug.value 中间解
                      → miss1≥6 且 miss2≥6 → 降级入库（ok_mode='debug'）
                      → 否则丢弃（ok_mode='fail'）
```

- P0 小批量（8条）验证结果：ok=2, debug降级=6, fail=0；共提取 667 对，每条约 83 步。
- θ 跨轨迹区分度良好（wT 从 ~1000 到 ~8 万；wE 从 10 到 1e4）。

### 1.4 关键调用链（最新）

```
samples_clean.mat
  → reconstruct_geometry
  → solve_warmstart (阶段A, 扫 wE, 输出 tau 含 stage 标签)
  → solve_ss_trajectory (阶段B, 符号 RK4 + IPOPT L-BFGS, 输出 theta_K 3×K)
  → extract_expert_transitions (数值 RK4 重跑, dt=0.1s, 输出 state 13D + action 4D)
  → generate_ps_expert_library 汇总 → expert_lib.mat
```

---

## 2. 核心架构与关键设计决策

1. **单打靶而非配点**：决策仅 3K=30 个变量（θ 分段常值），状态由符号 RK4 滚动积分保证动力学一致，无缺陷约束；配点方案 18 维刚性态 + 缺陷约束极难收敛（已实测放弃）。

2. **GAIL state 用相对态势量**（13D），不用绝对 18 维：原论文（Hui 2025）能用绝对量是因为"固定突防走廊 + 小扰动 + 2D"三件套成立；本项目 3D + 大范围随机初态，同一相对态势对应天差地别的绝对坐标，绝对量泛化差。

3. **J 三项固定系数**（`s_w=2e3, s_wE=55, s_wT=22`）经标定平衡数量级，避免目标项主导导致 miss 压到下限、θ 贴边退化。系数是常数评分，θ 是决策变量，两者严格分离。

4. **目标项用规避末端 ZEM**（`casadi_target_zem`），不用窗口内 r_MT 最小值：规避段攻弹离目标仍远，末端 ZEM 反映规避后能否对准目标（供 stage3 PN 收尾）。

5. **extract 重跑而非直接从 tau_warm 抽**：tau_warm 时间步 dt=1e-3s 过密（1000:1 冗余），用 dt=0.1s 重跑既控制规模又与 K 段 theta 分配对齐。

6. **expert_sa 列序（重要，质检时别查错列）**：

| 列 | 含义 |
|---|---|
| 1 | r_MD1（攻-D1距离） |
| 2 | rdot_MD1（接近率） |
| 3 | qy_MD1（LOS 高低角） |
| 4 | qz_MD1（LOS 方位角） |
| 5 | r_MD2 |
| 6 | rdot_MD2 |
| 7 | qy_MD2 |
| 8 | qz_MD2 |
| 9 | r_MT（攻-目标距离） |
| 10 | qy_MT |
| 11 | qz_MT |
| 12 | V_M（攻弹速度） |
| **13** | **tgo_norm**（归一化剩余时间，[0,1]） |
| **14** | **w**（恒=1e4） |
| **15** | **γ** |
| **16** | **wT** |
| **17** | **wE** |

---

## 3. 待办 TODO 与下一步计划

### 3.1 你需要自己跑的

- [ ] **P2（100条，约 35 分钟）**：验证管线质量

```matlab
cd('D:\LTY\yan\yanerxia\GAIL')
opts.run_stageB = true; opts.plot = false;
opts.save = true; opts.save_file = 'expert_lib_p2.mat';
opts.checkpoint_every = 20;
r100 = generate_ps_expert_library(100, opts);
```

- [ ] **P3（1000条，约 5.5 小时）**：正式批处理

```matlab
opts.save_file = 'expert_lib_1000.mat';
opts.checkpoint_every = 50;
r1000 = generate_ps_expert_library(1000, opts);
```

中断后读 checkpoint：

```matlab
ckpts = dir('expert_lib_1000_ckpt*.mat');
[~,idx] = max([ckpts.datenum]);
tmp = load(ckpts(idx).name,'results_ckpt');
r_partial = tmp.results_ckpt;
```

### 3.2 P2/P3 验证标准

**批处理统计（必看）**：

| 指标 | P2 标准 | P3 标准 |
|---|---|---|
| 阶段A成功率 | ≈ 100% | ≈ 100% |
| 阶段B可用率（ok+debug）/N | ≥ 90% | ≥ 80% |
| 平均每条步数 | 80～150 | 同左 |
| expert_sa 列数 | 17 | 17 |

**expert_sa sanity check（修正版，勿查错列）**：

```matlab
sa = r.expert_sa;
assert(size(sa,2) == 17, '列数不对')
assert(all(sa(:,14) == 1e4), 'w 应恒为 1e4')        % 第14列=w
assert(all(sa(:,13) >= -1e-6), 'tgo_norm 不应为负')  % 第13列=tgo_norm
assert(max(sa(:,13)) <= 1.001, 'tgo_norm 超出 [0,1]')
assert(all(sa(:,12) > 0), 'V_M 应 > 0')             % 第12列=V_M
assert(all(sa(:,17) >= 0), 'wE 不应为负')            % 第17列=wE
```

**θ 区分度（GAIL 关键）**：

```matlab
gamma_all=[]; wT_all=[]; wE_all=[];
for k=1:numel(r.trajectories)
    tr=r.trajectories(k);
    if tr.ok_use && ~isempty(tr.stageB) && ~isempty(tr.stageB.theta_K)
        tk=tr.stageB.theta_K;
        gamma_all=[gamma_all,tk(1,:)];
        wT_all=[wT_all,tk(2,:)];
        wE_all=[wE_all,tk(3,:)];
    end
end
fprintf('gamma: std=%.3f\nwT: std=%.1f\nwE: std=%.1f\n', ...
    std(gamma_all), std(wT_all), std(wE_all))
% 建议: std(wT)>1000, std(wE)>100
```

### 3.3 P3 完成后的后续任务

- [ ] **确认最终 `expert_lib_1000.mat` 格式**（17 列，约 8-15 万行，无 NaN）
- [ ] **与 GAIL 训练代码对接**：把 `expert_sa` 喂给判别器/生成器训练；state 13D → 网络输入，action 4D → 网络输出对齐（w 列可在训练时直接丢弃，或保留作校验）
- [ ] **DDPG 环境对接**：DDPG rollout 时每步用策略输出的 `[γ, wT, wE]`（加上固定 `w=1e4`）调用 `computeOptimalCmd_duo_target`（式47）得攻弹加速度，继续仿真
- [ ] **奖励函数设计**：见 `GAIL-DDPG_3D_Extension_Template.md` §7.2，核心三项：规避收益（增大 min 弹间距）/ 能耗惩罚 / 终端目标偏差

---

## 4. 避坑指南 / 已知问题

### 4.1 expert_sa 列号（本轮新增避坑）

- **`tgo_norm` 在第 13 列，不是第 17 列**；第 17 列是 `wE`，取值范围 [10, 1e4]。质检时若断言 `sa(:,17)<=1.001` 会必然报错。
- 列序：state 1-13，action 14-17（w/γ/wT/wE）。

### 4.2 γ bang-bang 是正常现象

γ 分段在 ±3 间切换是式47 闭环律的结构性结果（γ 对 miss 近单调），**不是退化**。跨轨迹 γ 分布虽然集中在边界，但 wT/wE 有足够区分度，GAIL 仍有有效信号。如需严格内点可加 γ 正则项，但会偏离纯式22。

### 4.3 IPOPT debug 降级（约 75% 条）

300 iter 上限下 IPOPT 多数返回中间解（非完全收敛），代码自动检查 miss≥6 后入库，物理可行性有保障。无需人工干预。若希望 ok 率更高可将 `opts.ss_cfg.max_iter` 设为 500，但每条额外增加约 5-10s。

### 4.4 CasADi 路径

`E:\matlab\casadi-3.7.2-windows64-matlab2018b`，`import casadi.*` 即可，`generate_ps_expert_library` 里 `add_casadi_path()` 已自动处理。

### 4.5 softmin 必须稳定化

`miss = vmin - beta*log(sum1(exp(-(v-vmin)/beta)))`，直接用 `exp(-v/beta)` 大距离时下溢 → `log(0)=-Inf` → 梯度 NaN。beta=3。

### 4.6 符号 RK4 Hessian 用 L-BFGS

决策仅 30 个，精确 Hessian 对长滚动图太慢；`hessian_approximation='limited-memory'` 是必须选项。

### 4.7 过载饱和做进动力学

`satf` 光滑饱和写在 `casadi_dynamics` 内，不能用硬约束 `‖aM‖≤8g`（与式47 冲突 → `Restoration_Failed`）。

### 4.8 批处理中断恢复

`checkpoint_every=50` 时每 50 条保存 `*_ckptXXXX.mat`；中断后 `expert_lib_1000.mat` 不存在，用最新 checkpoint 读 `results_ckpt`。**checkpoint 变量名是 `results_ckpt`，不是 `results`**，load 时注意。

---

## 5. 接续点（新对话从这里开始）

- **P2 和 P3 的运行命令**已在本文 §3.1 给出，你自己在 MATLAB 跑。
- **P3 完成后**第一件事：跑 §3.2 的质检脚本，确认 `expert_sa` 格式、可用率、θ 区分度均达标，再对接 GAIL 训练。
- **最终 GAIL 对接**参考 `GAIL-DDPG_3D_Extension_Template.md` §4（GAIL-DDPG 训练框架）和 §7（替代 BP 的实施方案）。
