param(
    [switch]$Smoke,
    [int]$Steps = 0,
    [string]$Seeds = "0,21,42",
    [switch]$SkipAnalysis
)

$ErrorActionPreference = "Stop"
$ExperimentDir = $PSScriptRoot
$MonthDir = Split-Path -Parent $ExperimentDir
$RepoRoot = Split-Path -Parent $MonthDir
$Python = "E:\Python\venvs\rl_pytorch\Scripts\python.exe"

if ($Steps -eq 0) {
    $Steps = if ($Smoke) { 1000 } else { 150000 }
}
$BufferSize = 300000
$EvalFreq = 25000
$NEval = 150
if ($Smoke) {
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

# `-File` 模式下 [int[]] 会把 "0,21" 按千分位解析成 21，故按字符串接收再逐个取数字
$Seeds = @($Seeds | Select-String -Pattern "\d+" -AllMatches |
    ForEach-Object { $_.Matches } | ForEach-Object { [int]$_.Value })
if ($Seeds.Count -eq 0) { throw "未解析到任何 seed" }

$Manifest = [ordered]@{
    experiment = "P0: pure online TD3 baseline under the P1 protocol (no expert Replay, no BC reg, no BC init)"
    created_at = (Get-Date).ToString("o")
    smoke = [bool]$Smoke
    seeds = $Seeds
    steps = $Steps
    eval_freq = $EvalFreq
    n_eval = $NEval
    eval_at_start = $true
    action_mapping = "ratio_log_box_v1"
    observation = "13D clean, identification disabled"
    entry = "td3_baseline/train_td3_pure.py (SB3 TD3, hyperparameters aligned with rl_td3/train_td3.py DEFAULTS)"
    reference = "P1_E1_BC_WARMSTART_20260927 (E0/E1) and P1_E1V_VANILLA_20260928 (E1V), same 150 fixed val ICs, eval_ic_seed=0"
    python = $Python
    git_commit = (& git rev-parse HEAD)
}
$ManifestName = "experiment_manifest_seeds_$($Seeds -join '_').json"
$Manifest | ConvertTo-Json -Depth 5 | Set-Content -Path (Join-Path $OutputRoot $ManifestName) -Encoding UTF8

foreach ($Seed in $Seeds) {
    $RunDir = Join-Path $OutputRoot "runs\seed_$Seed\TD3"
    $RunName = "p0_TD3_seed$Seed"
    $FinalModel = Join-Path $RunDir "$RunName.zip"
    if (Test-Path $FinalModel) {
        Write-Host "[runner] 跳过已完成：$FinalModel"
        continue
    }
    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
    $Td3Args = @(
        "td3_baseline/train_td3_pure.py",
        "--steps", "$Steps",
        "--buf", "$BufferSize",
        "--eval-freq", "$EvalFreq",
        "--n-eval", "$NEval",
        "--eval-ic-seed", "0",
        "--eval-at-start",
        "--save-dir", $RunDir,
        "--name", $RunName,
        "--seed", "$Seed",
        "--action-mapping", "ratio_log_box_v1",
        "--log-file", (Join-Path $RunDir "train.log")
    )
    Invoke-LoggedPython $Td3Args (Join-Path $RunDir "launcher.log")
}

if (-not $SkipAnalysis) {
    $AnalyzeArgs = @(
        (Join-Path $ExperimentDir "analyze_td3_pure.py"),
        "--root", $OutputRoot
    )
    Invoke-LoggedPython $AnalyzeArgs (Join-Path $OutputRoot "analysis.log")
}
