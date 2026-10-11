param(
    [switch]$Smoke,
    [int[]]$Seeds = @(0, 21, 42)
)

$ErrorActionPreference = "Stop"
$ExperimentDir = $PSScriptRoot
$RepoRoot = Split-Path -Parent $ExperimentDir
$Python = "E:\Python\venvs\rl_pytorch\Scripts\python.exe"
$OutputRoot = if ($Smoke) { Join-Path $ExperimentDir "_smoke" } else { $ExperimentDir }

if ($Smoke) {
    $Steps = 200
    $BufferSize = 4000
    $BcRatio = 1.0
    $BcBudget = 4000
    $BcEpochs = 3
    $EvalFreq = 100
    $NEval = 5
} else {
    $Steps = 300000
    $BufferSize = 300000
    $BcRatio = 0.33
    $BcBudget = 99000
    $BcEpochs = 80
    $EvalFreq = 25000
    $NEval = 150
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

$Manifest = [ordered]@{
    experiment = "P1/E1: Actor BC warm-start"
    created_at = (Get-Date).ToString("o")
    smoke = [bool]$Smoke
    seeds = $Seeds
    steps = $Steps
    eval_freq = $EvalFreq
    n_eval = $NEval
    action_mapping = "ratio_log_box_v1"
    observation = "13D clean, identification disabled"
    conditions = @(
        "E0: random Actor + expert Replay + decayed BC regularization",
        "E1: E0 + paired BC Actor/Actor-target warm-start"
    )
    python = $Python
    git_commit = (& git rev-parse HEAD)
}
$ManifestPath = Join-Path $OutputRoot "experiment_manifest.json"
if (-not (Test-Path $ManifestPath)) {
    $Manifest | ConvertTo-Json -Depth 5 | Set-Content `
        -Path $ManifestPath -Encoding UTF8
}

foreach ($Seed in $Seeds) {
    $SeedRoot = Join-Path $OutputRoot "runs\seed_$Seed"
    $BcDir = Join-Path $SeedRoot "bc"
    $BcBest = Join-Path $BcDir "bc_seed${Seed}_best.pt"
    New-Item -ItemType Directory -Force -Path $BcDir | Out-Null

    if (-not (Test-Path $BcBest)) {
        # 指定一个不存在的输入计划，强制每个 seed 按自己的 sample_seed 生成配对计划。
        $ForceBuildPlan = Join-Path $BcDir "_build_new_plan.npz"
        $BcArgs = @(
            "rl_td3/train_bc.py",
            "--sample-plan", $ForceBuildPlan,
            "--save-dir", $BcDir,
            "--name", "bc_seed$Seed",
            "--seed", "$Seed",
            "--sample-seed", "$Seed",
            "--budget", "$BcBudget",
            "--epochs", "$BcEpochs",
            "--actor-eligible", "both_ok",
            "--action-mapping", "ratio_log_box_v1",
            "--no-log"
        )
        Invoke-LoggedPython $BcArgs (Join-Path $BcDir "train_bc.log")
    }

    foreach ($Condition in @("E0", "E1")) {
        $RunDir = Join-Path $SeedRoot $Condition
        $RunName = "p1_${Condition}_seed$Seed"
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
            "--bc-ratio", "$BcRatio",
            "--eval-freq", "$EvalFreq",
            "--n-eval", "$NEval",
            "--eval-at-start",
            "--save-dir", $RunDir,
            "--name", $RunName,
            "--seed", "$Seed",
            "--sample-seed", "$Seed",
            "--save-sample-plan", (Join-Path $RunDir "expert_sample_plan.npz"),
            "--actor-eligible", "both_ok",
            "--action-mapping", "ratio_log_box_v1",
            "--log-file", (Join-Path $RunDir "train.log")
        )
        if ($Condition -eq "E1") {
            $Td3Args += @("--bc-init-ckpt", $BcBest)
        }
        Invoke-LoggedPython $Td3Args (Join-Path $RunDir "launcher.log")
    }
}

Invoke-LoggedPython @(
    (Join-Path $ExperimentDir "analyze_p1_e1.py"),
    "--root", $OutputRoot,
    "--budget", "$Steps"
) (Join-Path $OutputRoot "analysis.log")
