param(
 [string]$ModelDirectory='C:\Models\DirectAI\CyberRealistic-LCM-DirectAI',
 [string]$OutputDirectory='C:\dev\DirectAI\output\onnx-conversion-validation'
)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
[Reflection.Assembly]::LoadFrom((Join-Path $root 'projects\bin\DirectAI.ModelConversion\Release\net11.0\DirectAI.ModelConversion.dll')) | Out-Null
$manifest=Get-Content (Join-Path $ModelDirectory 'model.json') -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$checks=@(foreach($stage in $manifest.stages){
 if((Get-FileHash $stage.model).Hash -ne $stage.modelSha256){throw "Model hash mismatch: $($stage.stage)"}
 if((Get-FileHash $stage.weights).Hash -ne $stage.weightsSha256){throw "Weight file hash mismatch: $($stage.stage)"}
 $program=[DirectAI.ModelConversion.OnnxInspector]::GraphProgramSha256($stage.model)
 if($program -ne [DirectAI.ModelConversion.OnnxInspector]::GraphProgramSha256($stage.template)){throw "Graph program changed: $($stage.stage)"}
 $stream=[IO.File]::OpenRead($stage.weights)
 try{
  foreach($weight in $stage.sourceWeights){
   $stream.Position=$weight.offset
   $payload=[byte[]]::new([int]$weight.bytes)
   $stream.ReadExactly($payload)
   $hash=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($payload))
   if($hash -ne $weight.sha256){throw "Initializer payload mismatch: $($weight.initializer)"}
  }
 }finally{$stream.Dispose()}
 [pscustomobject]@{stage=$stage.stage;graphProgramSha256=$program;verifiedInitializers=$stage.boundInitializers;maximumTemplateDifference=($stage.sourceWeights.templateDifference.maximumAbsoluteError | Measure-Object -Maximum).Maximum}
})
$overwriteRejected=$false
try{[DirectAI.ModelConversion.CheckpointConverter]::Convert($manifest.checkpoint,$manifest.graphRecipeDirectory,$ModelDirectory,'sd15') | Out-Null}catch{$overwriteRejected=$_.Exception.ToString().Contains('new or empty')}
if(-not $overwriteRejected){throw 'Existing output rejection failed.'}
$sdxlRejected=$false
try{[DirectAI.ModelConversion.CheckpointConverter]::Convert($manifest.checkpoint,$manifest.graphRecipeDirectory,(Join-Path $OutputDirectory 'sdxl-rejection'),'sdxl') | Out-Null}catch{$sdxlRejected=$_.Exception.ToString().Contains('SDXL')}
if(-not $sdxlRejected){throw 'Unsupported architecture rejection failed.'}
$invalid=Join-Path $OutputDirectory 'invalid.safetensors'
[IO.File]::WriteAllBytes($invalid,[byte[]]::new(8))
$invalidRejected=$false
try{[DirectAI.ModelConversion.CheckpointConverter]::Convert($invalid,$manifest.graphRecipeDirectory,(Join-Path $OutputDirectory 'invalid-rejection'),'sd15') | Out-Null}catch{$invalidRejected=$_.Exception.ToString().Contains('Invalid safetensors header')}
if(-not $invalidRejected){throw 'Invalid checkpoint rejection failed.'}
$images=@(Get-ChildItem $OutputDirectory -Filter 'converted-single-111-*.png' | ForEach-Object {[pscustomobject]@{file=$_.Name;sha256=(Get-FileHash $_.FullName).Hash}})
$receipt=[ordered]@{createdUtc=[DateTime]::UtcNow.ToString('o');checkpointSha256=$manifest.checkpointSha256;sourceWeightCount=$manifest.sourceWeightCount;usedSourceWeightCount=$manifest.usedSourceWeightCount;unusedSourceWeights=$manifest.unusedSourceWeights;stages=$checks;existingOutputRejected=$overwriteRejected;unsupportedSdxlRejected=$sdxlRejected;invalidCheckpointRejected=$invalidRejected;images=$images;executionEvidence='Adjacent benchmark JSON records; integrity checks do not imply full numerical equivalence to a PyTorch source execution.'}
$receipt | ConvertTo-Json -Depth 12 | Set-Content (Join-Path $OutputDirectory 'conversion-validation.json')
$receipt
