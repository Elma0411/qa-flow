$ErrorActionPreference = 'Stop'

if (-not $env:HF_ENDPOINT) {
  $env:HF_ENDPOINT = 'https://hf-mirror.com'
}

$envName = 'pytorch'
$labelsFileName = [string]::Concat([char]0x4E09, [char]0x7EA7, [char]0x77E5, [char]0x8BC6, [char]0x6807, [char]0x7B7E, '.txt')
$labels = Join-Path 'qa/dataset' $labelsFileName

$datasetDir = 'runtime_assets/knowledge_tagging_3lvl/outputs/webhq_dataset_v1'
$modelDir = 'runtime_assets/knowledge_tagging_3lvl/outputs/model_rbt3_webhq_v1'

# 1) Build dataset: more "native" web text (gov.cn) + standards (openstd) + light synthetic for coverage.
conda run -n $envName python -m qa.knowledge_tagging_3lvl.scripts.build_dataset `
  --labels $labels `
  --out $datasetDir `
  --synth-per-label 10 `
  --crawl-openstd `
  --openstd-max-per-type 400 `
  --crawl-govcn `
  --govcn-max-pages 20 `
  --govcn-max-items 300 `
  --govcn-max-paragraphs 3 `
  --govcn-max-chars 1200

# 2) Train (new output dir; do NOT overwrite previous demo_model_rbt3).
conda run -n $envName python -m qa.knowledge_tagging_3lvl.scripts.train `
  --labels $labels `
  --train "$datasetDir/train.jsonl" `
  --val "$datasetDir/val.jsonl" `
  --out $modelDir `
  --device cuda `
  --amp `
  --epochs 6 `
  --batch-size 16 `
  --grad-accum-steps 2 `
  --lr 1e-5 `
  --max-length 192

# 3) Evaluate (full test + web-only test if available).
conda run -n $envName python -m qa.knowledge_tagging_3lvl.scripts.evaluate `
  --labels $labels `
  --model-dir $modelDir `
  --test "$datasetDir/test.jsonl" `
  --device cuda `
  --batch-size 64 `
  --by-source `
  --out "$modelDir/test_report.json"

if (Test-Path "$datasetDir/test_web.jsonl") {
  conda run -n $envName python -m qa.knowledge_tagging_3lvl.scripts.evaluate `
    --labels $labels `
    --model-dir $modelDir `
    --test "$datasetDir/test_web.jsonl" `
    --device cuda `
    --batch-size 64 `
    --by-source `
    --out "$modelDir/test_report_web.json"
}

Write-Host "Done."
Write-Host "Dataset: $datasetDir"
Write-Host "Model:   $modelDir"
Write-Host "Report:  $modelDir/test_report.json"
