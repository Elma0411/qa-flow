$ErrorActionPreference = 'Stop'

if (-not $env:HF_ENDPOINT) {
  $env:HF_ENDPOINT = 'https://hf-mirror.com'
}

$envName = 'pytorch'
$labelsFileName = [string]::Concat([char]0x4E09, [char]0x7EA7, [char]0x77E5, [char]0x8BC6, [char]0x6807, [char]0x7B7E, '.txt')
$labels = Join-Path 'qa/dataset' $labelsFileName
$datasetDir = 'runtime_assets/knowledge_tagging_3lvl/outputs/large_dataset'
$modelDir = 'runtime_assets/knowledge_tagging_3lvl/outputs/demo_model_rbt3'

# 1) Build a larger dataset (synthetic coverage + openstd national standards).
conda run -n $envName python -m qa.knowledge_tagging_3lvl.scripts.build_dataset `
  --labels $labels `
  --out $datasetDir `
  --synth-per-label 200 `
  --crawl-openstd `
  --openstd-max-per-type 2000

# 2) Continue training on GPU, overwriting weights in $modelDir.
conda run -n $envName python -m qa.knowledge_tagging_3lvl.scripts.train `
  --labels $labels `
  --train "$datasetDir/train.jsonl" `
  --val "$datasetDir/val.jsonl" `
  --out $modelDir `
  --resume-from $modelDir `
  --device cuda `
  --amp `
  --epochs 6 `
  --batch-size 16 `
  --grad-accum-steps 2 `
  --max-length 192

# 3) Evaluate.
conda run -n $envName python -m qa.knowledge_tagging_3lvl.scripts.evaluate `
  --labels $labels `
  --model-dir $modelDir `
  --test "$datasetDir/test.jsonl" `
  --device cuda `
  --batch-size 64 `
  --out "$modelDir/test_report.json"

Write-Host "Done."
Write-Host "Model: $modelDir"
Write-Host "Report: $modelDir/test_report.json"
