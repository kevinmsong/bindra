<#
.SYNOPSIS
  Train BINDRA-v2 and the analog generator, then evaluate both.

.DESCRIPTION
  Four stages, run in order, each logged to runs\logs\. The script STOPS at the
  first failure rather than carrying on, because every later stage depends on
  the one before it and a chain that continues past a failure wastes hours
  producing an artifact nobody can interpret.

  This is a LONG run. BINDRA-v1 took 2.79 hours an epoch at model_dim 96 on a
  single-layer cache; v2 is ~3.5x the parameters and reads 4x the protein bytes
  per batch, so budget several hours an epoch. With --patience 3 it will
  usually stop well before the 20-epoch budget, but plan in days not hours.

  Nothing here opens the sealed panels or the test partition. Both are
  one-way doors and neither should be touched until a model is finished.

.PARAMETER SkipAffinity
  Skip stage 1 and 2, e.g. to train only the generator.

.PARAMETER SkipGenerator
  Skip stage 3 and 4.

.PARAMETER ProteinCache
  Which protein cache to train against. The layer count is read from the cache
  itself, so a single-layer cache works here without any other change.

.EXAMPLE
  .\scripts\run_v2.ps1
.EXAMPLE
  .\scripts\run_v2.ps1 -ProteinCache data\cache\proteins -Tag v2-nolayers
#>
[CmdletBinding()]
param(
  [string]$Tag           = "v2",
  [string]$Regime        = "random_pair",
  [string]$ProteinCache  = "data\cache\proteins_l4",
  [string]$KnowledgeCache= "data\cache\knowledge",
  [string]$LigandCheckpoint = "runs\pretrain-random_pair-v1\best.pt",
  [int]   $ResidueBudget = 16384,
  [int]   $MaxPairs      = 96,
  [int]   $Accumulate    = 32,
  [int]   $Epochs        = 20,
  [int]   $Patience      = 3,
  [double]$LearningRate  = 1e-4,
  [switch]$SkipAffinity,
  [switch]$SkipGenerator
)

$ErrorActionPreference = "Stop"
$python = ".venv\Scripts\python.exe"
$logDir = "runs\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Invoke-Stage {
  param([string]$Name, [string[]]$Arguments)

  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $log   = Join-Path $logDir "$stamp-$Name.log"
  Write-Host ""
  Write-Host ("=" * 72)
  Write-Host "  $Name"
  Write-Host "  started $(Get-Date -Format 'HH:mm:ss')   log: $log"
  Write-Host ("=" * 72)

  $started = Get-Date
  # Tee so the run is both visible now and recoverable later. -u keeps the
  # child's output unbuffered, or progress only appears when it exits.
  & $python -u @Arguments 2>&1 | Tee-Object -FilePath $log
  $code = $LASTEXITCODE
  $elapsed = (Get-Date) - $started

  if ($code -ne 0) {
    Write-Host ""
    Write-Host "FAILED: $Name exited $code after $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Red
    Write-Host "        see $log"
    Write-Host "        later stages depend on this one and were NOT run."
    exit $code
  }
  Write-Host ""
  Write-Host "ok: $Name in $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
}

Write-Host "BINDRA-v2 pipeline"
Write-Host "  tag            $Tag"
Write-Host "  regime         $Regime"
Write-Host "  protein cache  $ProteinCache"
Write-Host "  epochs         $Epochs (patience $Patience, lr $LearningRate)"
Write-Host "  memory         residue-budget $ResidueBudget, max-pairs $MaxPairs, accumulate $Accumulate"
Write-Host ""
Write-Host "  If training still runs out of memory, the engine now REFUSES after 5"
Write-Host "  consecutive failures and names a budget to try. It no longer skips on,"
Write-Host "  which would finish the run on short proteins only."

# -- 1. affinity -------------------------------------------------------------
if (-not $SkipAffinity) {
  Invoke-Stage "01-train-affinity" @(
    "-m","bindra.cli","train",
    "--regime",$Regime,"--tag",$Tag,
    "--epochs",$Epochs,"--patience",$Patience,"--lr",$LearningRate,
    "--model-dim","256","--fusion-layers","2","--feedforward-multiplier","2",
    "--adapter-bottleneck","512","--pool-heads","4",
    "--protein-cache",$ProteinCache,
    "--assay-conditioning",
    "--ligand-checkpoint",$LigandCheckpoint,
    "--knowledge-cache",$KnowledgeCache,
    "--ranking-weight","0.1",
    # v1's defaults (65536 / 256 / 8) cascade into out-of-memory here: a
    # four-layer cache is 4x the protein bytes per residue and model_dim 256
    # is 2.7x the width. Accumulate rises in step so the EFFECTIVE batch is
    # unchanged, just assembled from smaller pieces.
    "--residue-budget",$ResidueBudget,"--max-pairs",$MaxPairs,
    "--accumulate",$Accumulate
  )

  $affinity = "runs\$Regime-$Tag\best.pt"
  if (-not (Test-Path $affinity)) {
    Write-Host "FAILED: training reported success but $affinity does not exist" -ForegroundColor Red
    exit 1
  }

  # -- 2. evaluate, val then cal --------------------------------------------
  foreach ($partition in @("val","cal")) {
    Invoke-Stage "02-eval-$partition" @(
      "-m","bindra.cli","evaluate",$affinity,
      "--regime",$Regime,"--partition",$partition,
      "--protein-cache",$ProteinCache,
      "--knowledge-cache",$KnowledgeCache,
      "--residue-budget",$ResidueBudget,"--max-pairs",$MaxPairs,
      "--bootstrap","1000",
      "--out","runs\eval-$Tag-$partition.json"
    )
  }
}

# -- 3. generator ------------------------------------------------------------
if (-not $SkipGenerator) {
  Invoke-Stage "03-train-generator" @(
    "-m","bindra.cli","generate","train",
    "--regime",$Regime,"--tag",$Tag,
    "--epochs",$Epochs,"--patience",$Patience,"--lr",$LearningRate,
    "--out","runs\generator"
  )

  $generator = "runs\generator\$Regime-$Tag\best.pt"
  if (-not (Test-Path $generator)) {
    Write-Host "FAILED: generator training reported success but $generator is missing" -ForegroundColor Red
    exit 1
  }

  # -- 4. sample, as a smoke check that it produces molecules ---------------
  Invoke-Stage "04-sample-generator" @(
    "-m","bindra.cli","generate","sample",$generator,
    "--seed","CC(=O)Nc1ccc(O)cc1","--endpoint","IC50","--delta","1.0",
    "-n","64","--out","runs\generated-$Tag.json"
  )
}

Write-Host ""
Write-Host ("=" * 72)
Write-Host "  all stages complete" -ForegroundColor Green
Write-Host ("=" * 72)
Write-Host "  affinity   runs\$Regime-$Tag\best.pt"
Write-Host "  metrics    runs\eval-$Tag-val.json, runs\eval-$Tag-cal.json"
Write-Host "  generator  runs\generator\$Regime-$Tag\best.pt"
Write-Host "  samples    runs\generated-$Tag.json"
Write-Host "  logs       $logDir"
Write-Host ""
Write-Host "  v1 for comparison: RMSE 1.2432 (1.220 to 1.268), concordance 0.730,"
Write-Host "  Spearman 0.637, AUROC 0.806, AUPRC 0.749 on a 0.424 baseline."
Write-Host ""
Write-Host "  The test partition and the sealed panels were NOT opened."
