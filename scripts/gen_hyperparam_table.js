const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        AlignmentType, BorderStyle, WidthType, VerticalAlign } = require('docx');
const fs = require('fs');
const path = require('path');

const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const topBorder = { style: BorderStyle.SINGLE, size: 12, color: "000000" };
const midBorder = { style: BorderStyle.SINGLE, size: 6, color: "000000" };
const bottomBorder = { style: BorderStyle.SINGLE, size: 12, color: "000000" };

const headerBorders = { top: topBorder, bottom: midBorder, left: noBorder, right: noBorder };
const bodyBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };
const lastBorders = { top: noBorder, bottom: bottomBorder, left: noBorder, right: noBorder };

const TABLE_W = 9026;
const COLS = [3600, 2713, 2713];

function cellPara(text, { bold = false, center = true } = {}) {
  // Chinese + ASCII mixed: use SimSun for reliability in CN Word
  const hasCJK = /[\u4e00-\u9fff]/.test(text);
  return new Paragraph({
    alignment: center ? AlignmentType.CENTER : AlignmentType.LEFT,
    children: [
      new TextRun({
        text,
        bold,
        font: hasCJK ? "SimSun" : "Times New Roman",
        size: 21,
      }),
    ],
  });
}

function makeCell(text, borders, opts = {}) {
  return new TableCell({
    borders,
    width: { size: opts.width, type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 60, bottom: 60, left: 80, right: 80 },
    children: [cellPara(text, { bold: opts.bold, center: opts.center !== false })],
  });
}

function makeRow(cells, borders, widths = COLS) {
  return new TableRow({
    children: cells.map((c, i) => {
      const text = typeof c === "string" ? c : c.text;
      const bold = typeof c === "object" ? !!c.bold : false;
      const center = typeof c === "object" && c.center !== undefined ? c.center : true;
      return makeCell(text, borders, { width: widths[i], bold, center });
    }),
  });
}

const rows = [
  ["总训练步数", "300000", "300000"],
  ["学习率", "1e-4", "1e-4"],
  ["批次大小", "256", "256"],
  ["折扣因子 gamma", "0.99", "0.99"],
  ["软更新系数 tau", "0.005", "0.005"],
  ["经验回放容量", "300000", "300000"],
  ["开始更新步数", "1000", "0 (BC预填充后)"],
  ["训练频率", "1 step", "1 step"],
  ["策略延迟更新 policy_delay", "3", "3"],
  ["网络结构", "[256, 256]", "[256, 256]"],
  ["探索噪声 sigma", "0.1", "0.1"],
  ["目标策略噪声 target_policy_noise", "0.2", "0.2"],
  ["目标噪声裁剪 target_noise_clip", "0.5", "0.5"],
  ["梯度裁剪", "+/-10", "+/-10"],
  ["BC预填充比例", "-", "0.33"],
  ["BC正则初始系数 lambda0", "-", "1.0"],
  ["BC衰减步数", "-", "200000"],
];

const tableRows = [
  makeRow(
    [
      { text: "超参数", bold: true },
      { text: "TD3", bold: true },
      { text: "BC-TD3", bold: true },
    ],
    headerBorders
  ),
  ...rows.map((r, i) =>
    makeRow(
      r.map((t, j) => ({ text: t, bold: j === 0, center: j !== 0 })),
      i === rows.length - 1 ? lastBorders : bodyBorders
    )
  ),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Times New Roman", size: 21 } } },
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      children: [
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
          children: [
            new TextRun({
              text: "表1  TD3与BC-TD3训练超参数",
              bold: true,
              font: "SimSun",
              size: 21,
            }),
          ],
        }),
        new Table({
          width: { size: TABLE_W, type: WidthType.DXA },
          columnWidths: COLS,
          rows: tableRows,
        }),
        new Paragraph({
          spacing: { before: 200 },
          children: [
            new TextRun({
              text: "注：目标策略噪声与目标噪声裁剪为Stable-Baselines3 TD3默认值（训练脚本未显式覆盖）。探索噪声为在线交互时加在动作上的高斯噪声(sigma=0.1)。",
              font: "SimSun",
              size: 18,
            }),
          ],
        }),
        new Paragraph({
          spacing: { before: 80 },
          children: [
            new TextRun({
              text: "数据来源：td3_baseline/train_td3_pure.py（TD3）、rl_td3/train_td3.py（BC-TD3）。",
              font: "SimSun",
              size: 18,
            }),
          ],
        }),
      ],
    },
  ],
});

const outPath = path.join(__dirname, "..", "TD3_BCTD3_hyperparameters_v2.docx");
Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(outPath, buffer);
  console.log("Wrote:", outPath);
});
