import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
} from "cursor/canvas";

export default function BCTD3AdvantageExperimentDesign() {
  return (
    <Stack gap={22} style={{ padding: 20, maxWidth: 1180 }}>
      <Stack gap={8}>
        <H1>BC-TD3：现有优势、差距原因与强化方案</H1>
        <Text tone="secondary">
          只分析非辨识结果 · 最新公平对照：300k、seed=0、同一300个test初值
        </Text>
        <Row gap={8} wrap>
          <Pill active>重点：训练稳定性</Pill>
          <Pill>重点：BC预训练是否真正生效</Pill>
          <Pill>不把1.3 pp包装成显著优势</Pill>
        </Row>
      </Stack>

      <Callout tone="info" title="核心判断">
        当前 BC-TD3 最明显的优势不是 best_model 的最终成功率，而是中后期更不容易退化。
        现有流程并未真正执行“先BC预训练Actor、再TD3微调”，且专家作用在训练后期被逐步
        清除，因此当前小差距不能代表完整BC预训练方案的上限。
      </Callout>

      <Grid columns={4} gap={14}>
        <Stat value="+10.7 pp" label="300k final固定验证成功代理" tone="success" />
        <Stat value="−1.3 pp" label="BC best→final退化" tone="success" />
        <Stat value="−10.7 pp" label="Pure best→final退化" tone="danger" />
        <Stat value="+1.3 pp" label="best_model全量test差值" tone="warning" />
      </Grid>

      <Grid columns="1.1fr 0.9fr" gap={20} align="start">
        <Stack gap={8}>
          <H2>当前优势集中在哪里</H2>
          <Text tone="tertiary" size="small">
            横轴：比较口径；纵轴：联合成功率或其固定验证集奖励阈值代理（%）。
          </Text>
          <BarChart
            categories={["best全量test", "300k final固定val", "200k–300k平均val"]}
            series={[
              { name: "BC-TD3", data: [94.3, 96.0, 95.3], tone: "success" },
              { name: "Pure TD3", data: [93.0, 85.3, 90.9], tone: "info" },
            ]}
            height={245}
            beginAtZero={false}
            yMin={80}
            yMax={100}
            valueSuffix="%"
            showValues
          />
          <Text tone="secondary" size="small">
            来源：两组training_300k_20260730_1917的evaluations.npz与
            mc_fulltest_*_best_20260731.npz。
          </Text>
        </Stack>

        <Card>
          <CardHeader trailing={<Pill size="sm">现有证据</Pill>}>
            可以说与暂时不能说
          </CardHeader>
          <CardBody>
            <Stack gap={9}>
              <Text weight="semibold">可以说</Text>
              <Text>BC路径在200k之后更稳定，best到final的性能保持明显更好。</Text>
              <Text>final验证奖励标准差32.45，对比Pure 58.93，低约45%。</Text>
              <Text>test安全距离第10百分位9.31m，对比6.69m，存在尾部改善信号。</Text>
              <Divider />
              <Text weight="semibold">暂时不能说</Text>
              <Text>best全量test仅94.3%对93.0%，配对p=0.585。</Text>
              <Text>当前前150k并未更快收敛；典型安全距离和目标精度也无明显优势。</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <H2>尚未验证、但最可能出现明显优势的指标</H2>
      <Table
        headers={["优先级", "指标", "为什么可能体现BC优势", "怎样验证"]}
        rows={[
          [
            "1",
            "跨seed训练稳定性",
            "现有最大信号就是Pure后期回退、BC保持",
            "≥5 seeds；比较best→final降幅、最差checkpoint和崩溃概率",
          ],
          [
            "2",
            "能量与执行机构负担",
            "打靶目标含能量项，而当前RL奖励几乎未奖励节能",
            "比较∫||a||²dt、8g饱和率、动作总变差和尾部CVaR",
          ],
          [
            "3",
            "训练过程安全成本",
            "有效BC应减少早期随机探索造成的拦截",
            "统计达到90%成功前的累计拦截局数和环境交互步数",
          ],
          [
            "4",
            "困难耦合场景",
            "专家NLP显式处理双NCV与目标间耦合",
            "按|Δtgo|、b向量耦合、矩阵条件数和6g/7g/8g分层",
          ],
          [
            "5",
            "真正BC初始化后的低预算性能",
            "当前没有真实BC热启动，不能据现曲线否定其样本效率",
            "比较0、10k、25k、50k、100k步的成功率和学习曲线AUC",
          ],
        ]}
        rowTone={["success", "success", "info", "info", "warning"]}
        striped
      />

      <Divider />

      <H2>为什么当前最终差距较小</H2>
      <Table
        headers={["主要原因", "当前机制", "直接后果"]}
        rows={[
          [
            "没有真正BC预训练",
            "随机Actor从第0步同时接收随机Critic的Q梯度和BC正则",
            "专家策略尚未建立就可能被不可靠Q梯度带偏",
          ],
          [
            "BC相对Q项过弱",
            "λ×BC loss约1，而actor/Q项很快达到10–20以上",
            "名义λ接近1，实际专家约束仍迅速边缘化",
          ],
          [
            "专家影响后期消失",
            "λ在200k归零；约205k开始覆盖demo，300k时原始demo基本被覆盖",
            "最终模型主要按纯TD3继续演化，只保留路径依赖",
          ],
          [
            "状态与动作标签不可辨识",
            "13D状态缺qdot/姿态；γ多模态；8g饱和使多组权重物理等价",
            "动作MSE难学，且低MSE不等于正确闭环控制",
          ],
          [
            "任务和奖励存在天花板",
            "Pure在75k已约95%；成功奖励不区分能量、平滑性和过度规避",
            "纯TD3找到简单边界策略后即可接近专家成功率",
          ],
          [
            "专家质量与覆盖有限",
            "实际1287轨、812条训练both_ok，与论文2400轨不一致",
            "专家流形覆盖不足，困难区和策略偏离状态缺少标签",
          ],
        ]}
        rowTone={["danger", "danger", "warning", "warning", "info", "info"]}
        striped
      />

      <H2>最能强化并凸显BC预训练作用的方案</H2>
      <Grid columns={2} gap={18}>
        <Card>
          <CardHeader trailing={<Pill size="sm">先让BC可学</Pill>}>
            阶段一：建立有效Actor初值
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>1. 用当前67,740条both_ok标签重跑纯BC诊断。</Text>
              <Text>2. 补充LOS角速度及必要姿态/相对速度，降低同状态异动作。</Text>
              <Text>3. 在动作MSE外加入加速度aM或一步next_state一致性损失。</Text>
              <Text>4. 对γ采用符号/模式头，避免把±边界平均到0。</Text>
              <Text>5. 以闭环val成功率选BC best，而不是只看离线MSE。</Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={<Pill size="sm">再保护专家作用</Pill>}>
            阶段二：正确接入TD3
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>1. 同步加载BC best到actor与actor_target。</Text>
              <Text>2. 先冻结Actor，用demo预热Critic，再解冻联合微调。</Text>
              <Text>3. 使用不可覆盖的独立demo buffer，每批固定采20%–30%。</Text>
              <Text>4. 按Q值或梯度尺度归一化BC项，并保留λ下限或使用Q-filter。</Text>
              <Text>5. 用DAgger补充策略偏离专家轨迹后访问的状态。</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <H2>最小因果对照</H2>
      <Table
        headers={["组别", "配置", "回答的问题"]}
        rows={[
          ["A", "Pure TD3", "无专家基线"],
          ["B", "当前：随机Actor + demo预填 + 衰减BC", "现有实现的增益"],
          ["C", "BC best初始化 + 普通TD3", "纯BC热启动是否有效"],
          ["D", "BC初始化 + 固定demo采样 + 自适应BC", "完整方案能否形成持续优势"],
        ]}
        rowTone={[undefined, "warning", "info", "success"]}
        striped
      />

      <Callout tone="warning" title="论文主张应优先落在这三项">
        若完整方案成立，最应展示的是：低在线步数下更高成功率、训练后期更少性能退化、
        以及同等成功率下更低能耗/饱和率。它们比在94%平台上追求1–2个百分点更能说明
        打靶专家轨迹和BC预训练的独立价值。
      </Callout>

      <Text tone="tertiary" size="small">
        数据范围：非辨识结果；参数辨识仅保留理论扩展方向，未使用项目辨识实验数据。
      </Text>
    </Stack>
  );
}
