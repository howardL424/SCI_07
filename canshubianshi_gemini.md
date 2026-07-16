在这篇论文中，状态空间方程（公式11）是为了将非线性的相对运动动力学转化为“**局部线性时变（但每个采样周期内近似定常）**”的系统。这为应用最小二乘法（Least Squares, LS）进行在线参数辨识提供了非常好的数学基础。

下面我将为你推导如何将公式(11)转化为标准最小二乘法的计算公式。

### 1. 提取待辨识的非零元素
根据公式(11)和上下文的定义：
*   **对于矩阵 $\mathbf{A}_i$**：第一行是 $[0, 1]$（纯运动学积分关系，不需要辨识）；需要辨识的非零元素在第二行第一列，我们记为 $A_{21} = \dot{q}_{yi}^2 + \dot{q}_{zi}^2 \cos^2 q_{yi}$。
*   **对于矩阵 $\mathbf{B}_i$**：需要辨识的非零元素是第二行的 $B_{1i}$ 和 $B_{2i}$。
*   **对于矩阵 $\mathbf{C}_i$**：包含防御者的机动加速度 $C_{1i}$（通常视为外部扰动或已知项，如果是完全非合作的，可以把它当作一个常数偏置项一起辨识）。

将公式(11)的矩阵乘法展开，实质上只需要看**第二行（加速度层面的方程）**：
$$ \ddot{r}_{ADi}(t) = A_{21} \cdot r_{ADi}(t) + B_{1i} \cdot a_{yA}(t) + B_{2i} \cdot a_{zA}(t) + C_{1i}(t) $$

---

### 2. 构建最小二乘的标准回归形式
最小二乘法的核心思想是将上述方程写成标准的线性回归模型：**$y(t) = \boldsymbol{\phi}^T(t) \boldsymbol{\theta}$**

这里分为两种情况，取决于你是否能通过观测设备估算出 $C_{1i}$：

#### 情况 A：假设 $C_{1i}$ 是可估算的（或作为已知扰动剔除）
令观测向量（系统的输出响应）为：
$$ y(t) = \ddot{r}_{ADi}(t) - C_{1i}(t) $$
*注：实际工程中 $\ddot{r}_{ADi}$ 可以通过相对速度 $\dot{r}_{ADi}$ 的差分近似得到，或由导引头和惯导信息解算。*

定义待辨识参数矩阵（列向量） $\boldsymbol{\theta}$：
$$ \boldsymbol{\theta} = \begin{bmatrix} A_{21} \\ B_{1i} \\ B_{2i} \end{bmatrix} $$

定义回归向量（系统的状态输入特征） $\boldsymbol{\phi}(t)$：
$$ \boldsymbol{\phi}(t) = \begin{bmatrix} r_{ADi}(t) \\ a_{yA}(t) \\ a_{zA}(t) \end{bmatrix} $$

#### 情况 B：假设 $C_{1i}$ 无法直接估算，需一并辨识（推荐真实场景使用）
将 $C_{1i}$ 视为短时间内的常数偏置，直接加入辨识向量：
$$ y(t) = \ddot{r}_{ADi}(t) $$
$$ \boldsymbol{\theta} = \begin{bmatrix} A_{21} \\ B_{1i} \\ B_{2i} \\ C_{1i} \end{bmatrix}, \quad \boldsymbol{\phi}(t) = \begin{bmatrix} r_{ADi}(t) \\ a_{yA}(t) \\ a_{zA}(t) \\ 1 \end{bmatrix} $$

---

### 3. 最小二乘公式的实现

在制导控制系统中，通常使用离散的采样点。假设你在时间步 $k=1, 2, ..., N$ 收集了数据。

#### 方法一：批处理最小二乘法 (Batch LS) - 适合预先分析或短窗口滑动
如果你积累了 $N$ 个时间步的数据，可以构造矩阵：
$$ \mathbf{Y} = \begin{bmatrix} y(1) \\ y(2) \\ \vdots \\ y(N) \end{bmatrix}, \quad \boldsymbol{\Phi} = \begin{bmatrix} \boldsymbol{\phi}^T(1) \\ \boldsymbol{\phi}^T(2) \\ \vdots \\ \boldsymbol{\phi}^T(N) \end{bmatrix} $$

最优参数的闭式解公式为：
$$ \hat{\boldsymbol{\theta}} = (\boldsymbol{\Phi}^T \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^T \mathbf{Y} $$

#### 方法二：带遗忘因子的递推最小二乘法 (FF-RLS) - 最适合论文中的在线制导
论文原文指出：*“该系统为线性时变系统，但在每个离散制导周期内可近似为定常系统。”*
既然本质是时变的（视线角不断变化导致参数变化），必须使用**带遗忘因子 $\lambda$ (通常取0.95~0.99) 的 RLS**，公式如下：

在每个控制周期 $k$，当你获得新的观测值 $y(k)$ 和状态 $\boldsymbol{\phi}(k)$ 时，执行以下三步进行在线迭代更新：

1. **计算增益矩阵 $\mathbf{K}(k)$：**
$$ \mathbf{K}(k) = \frac{\mathbf{P}(k-1)\boldsymbol{\phi}(k)}{\lambda + \boldsymbol{\phi}^T(k)\mathbf{P}(k-1)\boldsymbol{\phi}(k)} $$
2. **更新待辨识参数 $\hat{\boldsymbol{\theta}}(k)$：**
$$ \hat{\boldsymbol{\theta}}(k) = \hat{\boldsymbol{\theta}}(k-1) + \mathbf{K}(k) \left[ y(k) - \boldsymbol{\phi}^T(k)\hat{\boldsymbol{\theta}}(k-1) \right] $$
3. **更新协方差矩阵 $\mathbf{P}(k)$：**
$$ \mathbf{P}(k) = \frac{1}{\lambda} \left[ \mathbf{I} - \mathbf{K}(k)\boldsymbol{\phi}^T(k) \right] \mathbf{P}(k-1) $$

*(初始化：$\hat{\boldsymbol{\theta}}(0)$ 可设为零向量或代入理论初值，$\mathbf{P}(0)$ 设为一个较大的对角阵，如 $\alpha \mathbf{I}$, $\alpha=10^4$)*

### 💡 你的研究思路非常有潜力
原文是通过公式(12)直接用解析法算出的矩阵 A 和 B。但正如你注意到的，实际空战中 $q_{yi}, \theta_A, \psi_{V_A}$ 等角度极容易受测量噪声污染，纯代数计算会引发剧烈抖动。**如果你在你的框架中，利用在线RLS直接从加速度和距离的映射中“数据驱动”地辨识出这些非零元素，代替原来依赖精确视线角测量的解析公式，这将在鲁棒性上是一个极好的创新点。**