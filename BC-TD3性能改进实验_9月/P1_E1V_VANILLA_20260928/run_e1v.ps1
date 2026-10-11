param(
    [switch]$Smoke,
    [int]$Steps = 0,
    [int[]]$Seeds = @(0, 42),
    [string]$ExperimentName = "P1_E1V_VANILLA_20260928"
)

$ErrorActionPreference = "Stop"
$ExperimentDir = $PSScriptRoot
$RepoRoot = Split-Path -Parent $ExperimentDir
$Python = "E:\Python\venvs\rl_pytorch\Scripts\python.exe"
$P1Root = Join-Path $RepoRoot "P1_E1_BC_WARMSTART_20260927"

# 默认预算：冒烟 1000；预筛 50000；不指定 Steps 时按模式推断
if ($Steps -eq 0) {
    $Steps = if ($Smoke) { 1000 } else { 50000 }
}
$BufferSize = 300000
$EvalFreq = 25000
$NEval = 150
if ($Smoke) {
    # 冒烟只验证代码路径：评估更密集但保持同样的 150 个固定初态池逻辑
    $EvalFreq = 500
    $NEval = 5
}

$OutputRoot = $ExperimentDir
if ($Smoke) {
    $OutputRoot = Join-Path $ExperimentDir "_smoke"
}

function Invoke-LoggedPython {
    param(
        [string[]]$Arguments,
        [string]$LogPath
    )
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
    $PreviousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
    $ExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorAction
    if ($ExitCode -ne 0) {
        throw "命令失败（exit=$ExitCode）：$Python $($Arguments -join ' ')"
    }
}

Set-Location $RepoRoot
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

# 兼容 `powershell -File run_e1v.ps1 -Seeds @(0,42)` 被按字面量传入单字符串的情形
if ($Seeds.Count -eq 1 -and ("$($Seeds[0])" -match "[,;@\(\)]")) {
    $Parsed = "$($Seeds[0])" | Select-String -Pattern "\d+" -AllMatches |
        ForEach-Object { $_.Matches } | ForEach-Object { [int]$_.Value }
    if ($Parsed.Count -gt 0) { $Seeds = @($Parsed) }
}

$Manifest = [ordered]@{
    experiment = "P1/E1V: vanilla BC warm-start + standard TD3 (no expert Replay, no online BC reg)"
    created_at = (Get-Date).ToString("o")
    smoke = [bool]$Smoke
    seeds = $Seeds
    steps = $Steps
    eval_freq = $EvalFreq
    n_eval = $NEval
    action_mapping = "ratio_log_box_v1"
    observation = "13D clean, identification disabled"
    condition = "E1V: paired BC ckpt -> Actor/Actor-target init, --no-bc, --bc-lambda0 0, learning_starts=1000, random Critic"
    reference = "P1_E1_BC_WARMSTART_20260927 (E0/E1, same 150 fixed val ICs, eval_ic_seed=0)"
    python = $Python
    git_commit = (& git rev-parse HEAD)
}
$ManifestPath = Join-Path $OutputRoot "experiment_manifest.json"
$Manifest | ConvertTo-Json -Depth 5 | Set-Content -Path $ManifestPath -Encoding UTF8

foreach ($Seed in $Seeds) {
    $BcBest = Join-Path $P1Root "runs\seed_$Seed\bc\bc_seed${Seed}_best.pt"
    if (-not (Test-Path $BcBest)) {
        throw "缺少配对 BC checkpoint：$BcBest（E1V 必须按 seed 配对，不可混用）"
    }
    $SeedRoot = Join-Path $OutputRoot "runs\seed_$Seed"
    $RunDir = Join-Path $SeedRoot "E1V"
    $RunName = "p1_E1V_seed$Seed"
    $FinalModel = Join-Path $RunDir "$RunName.zip"
    if (Test-Path $FinalModel) {
        Write-Host "[runner] 跳过已完成：$FinalModel"
        continue
    }
    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
    $Td3Args = @(
        "rl_td3/train_td3.py",
        "--steps", "$Steps",
        "--buf", "$BufferSize",
        "--no-bc",
        "--bc-lambda0", "0",
        "--bc-init-ckpt", $BcBest,
        "--eval-freq", "$EvalFreq",
        "--n-eval", "$NEval",
        "--eval-at-start",
        "--save-dir", $RunDir,
        "--name", $RunName,
        "--seed", "$Seed",
        "--sample-seed", "$Seed",
        "--actor-eligible", "both_ok",
        "--action-mapping", "ratio_log_box_v1",
        "--log-file", (Join-Path $RunDir "train.log")
    )
    Invoke-LoggedPython $Td3Args (Join-Path $RunDir "launcher.log")
}

# 汇总 E1V 并与 P1 的 E0/E1 同评估点配对比较
$AnalyzeArgs = @(
    (Join-Path $ExperimentDir "analyze_e1v.py"),
    "--root", $OutputRoot,
    "--p1-root", $P1Root,
    "--budget", "$Steps"
)
Invoke-LoggedPython $AnalyzeArgs (Join-Path $OutputRoot "analysis.log")
