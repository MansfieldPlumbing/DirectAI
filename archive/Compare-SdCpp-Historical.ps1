param(
 [string]$Model='C:\Models\sd.cpp\Fluently-v4-LCM-f16.gguf',
 [string]$OnnxModel='C:\Models\DirectAI\Fluently-v4-LCM-amuse',
 [string]$OutputDirectory='C:\dev\DirectAI\output\sd15-head-to-head',
 [int]$Repeats=3,
 [switch]$DirectAIOnly,
 [switch]$SdCppOnly
)
$ErrorActionPreference='Stop'
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$prompt='a photograph of a red rose in a glass vase, natural daylight'
$request=@{prompt=$prompt;negative_prompt='';steps=6;seed=42;width=512;height=512;cfg_scale=1;sampler_name='LCM';scheduler='lcm';batch_size=1;n_iter=1}
$rows=[Collections.Generic.List[object]]::new()
if($SdCppOnly -and (Test-Path "$OutputDirectory\measurements.json")){
 foreach($row in (Get-Content "$OutputDirectory\measurements.json" -Raw | ConvertFrom-Json)){
  if($row.runtime -eq 'DirectAI ORT DirectML'){$rows.Add($row)}
 }
}
$env:DIRECTAI_OUTPUT_DIR=$OutputDirectory
if(-not $SdCppOnly){
 foreach($placement in @(
  @{label='directai-single-000';te=0;unet=0;vae=0},
  @{label='directai-single-111';te=1;unet=1;vae=1},
  @{label='directai-stages-011';te=0;unet=1;vae=1},
  @{label='directai-stages-012';te=0;unet=1;vae=2}
 )){
  $arguments=@('benchmark','--model',$OnnxModel,'--label',$placement.label,'--device-te',"$($placement.te)",'--device-unet',"$($placement.unet)",'--device-vae',"$($placement.vae)",'--steps','6','--seed','42','--width','512','--height','512','--warmups','1','--repeats',"$Repeats")
  & C:\dev\DirectAI\bin\Release\net11.0\DirectAI.exe @arguments > "$OutputDirectory\$($placement.label).log" 2>&1
  if($LASTEXITCODE -ne 0){throw "DirectAI failed: $($placement.label)"}
  foreach($file in Get-ChildItem $OutputDirectory -Filter "$($placement.label)-*.json"){
   $sidecar=Get-Content $file.FullName -Raw | ConvertFrom-Json
   $rows.Add([pscustomobject]@{runtime='DirectAI ORT DirectML';placement=$placement.label;iteration=$sidecar.metadata.iteration;inferenceMilliseconds=$sidecar.metadata.inferenceMilliseconds;metadata=$file.Name})
  }
  Write-Host "Completed $($placement.label)"
 }
}
if(-not $DirectAIOnly){
 if(-not(Test-Path -LiteralPath $Model)){throw "GGUF missing: $Model"}
 foreach($placement in @(
  @{label='sdcpp-single-000';backend='te=Vulkan0,diffusion=Vulkan0,vae=Vulkan0';extra=@()},
  @{label='sdcpp-single-111';backend='te=Vulkan1,diffusion=Vulkan1,vae=Vulkan1';extra=@()},
  @{label='sdcpp-stages-011';backend='te=Vulkan0,diffusion=Vulkan1,vae=Vulkan1';extra=@()},
  @{label='sdcpp-stages-012';backend='te=Vulkan0,diffusion=Vulkan1,vae=Vulkan2';extra=@()},
  @{label='sdcpp-stages-101';backend='te=Vulkan1,diffusion=Vulkan0,vae=Vulkan1';extra=@()},
  @{label='sdcpp-unet-two-default';backend='te=Vulkan1,diffusion=Vulkan0&Vulkan1,vae=Vulkan1';extra=@('--split-mode','layer')},
  @{label='sdcpp-unet-two-budget';backend='te=Vulkan1,diffusion=Vulkan0&Vulkan1,vae=Vulkan1';extra=@('--split-mode','layer','--max-vram','Vulkan0=1,Vulkan1=7')}
 )){
  $stdout="$OutputDirectory\$($placement.label).stdout.log"
  $stderr="$OutputDirectory\$($placement.label).stderr.log"
  $arguments=@('-m',$Model,'--backend',$placement.backend,'--auto-fit','off','--listen-port','12345','--conditioning-cache-size','0','--type','f16','--rng','cpu','--sampling-method','lcm','--scheduler','lcm','--cfg-scale','1')+$placement.extra
  $process=Start-Process -FilePath C:\bin\sd.cpp\sd-server.exe -ArgumentList $arguments -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
  try{
   $ready=$false
   for($attempt=0;$attempt -lt 180;$attempt++){
    $process.Refresh()
    if($process.HasExited){throw "sd-server exited $($process.ExitCode): $($placement.label). See $stderr"}
    try{Invoke-RestMethod 'http://127.0.0.1:12345/sdapi/v1/options' -TimeoutSec 1 | Out-Null;$ready=$true;break}catch{Start-Sleep -Milliseconds 500}
   }
   if(-not $ready){throw 'sd-server startup timeout'}
   for($iteration=-1;$iteration -lt $Repeats;$iteration++){
    $before=(Get-Content $stdout -ErrorAction SilentlyContinue).Count
    $timer=[Diagnostics.Stopwatch]::StartNew()
    $response=Invoke-RestMethod 'http://127.0.0.1:12345/sdapi/v1/txt2img' -Method Post -ContentType 'application/json' -Body ($request | ConvertTo-Json -Compress) -TimeoutSec 300
    $timer.Stop()
    if(-not $response.images -or $response.images.Count -ne 1){throw 'Expected exactly one image'}
    $lines=@(Get-Content $stdout | Select-Object -Skip $before)
    if($iteration -lt 0){continue}
    $imagePath="$OutputDirectory\$($placement.label)-$('{0:D2}' -f $iteration).png"
    [IO.File]::WriteAllBytes($imagePath,[Convert]::FromBase64String($response.images[0]))
    $match=[regex]::Matches(($lines -join [Environment]::NewLine),'generate_image completed in ([0-9.]+)s')
    $nativeMs=if($match.Count){[double]::Parse($match[$match.Count-1].Groups[1].Value,[Globalization.CultureInfo]::InvariantCulture)*1000}else{$null}
    $metadata=[ordered]@{schemaVersion=1;createdUtc=[DateTime]::UtcNow.ToString('o');output=[IO.Path]::GetFileName($imagePath);sha256=(Get-FileHash $imagePath).Hash;runtime='sd.cpp Vulkan';commit='3f8527a';model=$Model;backend=$placement.backend;arguments=$arguments;request=$request;iteration=$iteration;nativeInferenceMilliseconds=$nativeMs;httpRoundTripMilliseconds=$timer.Elapsed.TotalMilliseconds;nativeTimingResolutionMilliseconds=10;responseInfo=$response.info;requestLog=$lines}
    $metadataPath=[IO.Path]::ChangeExtension($imagePath,'.json')
    $metadata | ConvertTo-Json -Depth 12 | Set-Content $metadataPath
    $rows.Add([pscustomobject]@{runtime='sd.cpp Vulkan';placement=$placement.label;iteration=$iteration;inferenceMilliseconds=$nativeMs;httpMilliseconds=$timer.Elapsed.TotalMilliseconds;metadata=[IO.Path]::GetFileName($metadataPath)})
   }
   Write-Host "Completed $($placement.label)"
  }catch{
   @{placement=$placement.label;backend=$placement.backend;arguments=$arguments;error=$_.Exception.Message} | ConvertTo-Json -Depth 5 | Set-Content "$OutputDirectory\$($placement.label).error.json"
   Write-Warning "$($placement.label): $($_.Exception.Message)"
  }finally{
   $process.Refresh()
   if(-not $process.HasExited){Stop-Process -Id $process.Id; $process.WaitForExit()}
  }
 }
}
$rows | ConvertTo-Json -Depth 8 | Set-Content "$OutputDirectory\measurements.json"
$summary=@(foreach($group in $rows | Group-Object placement){
 $values=@($group.Group.inferenceMilliseconds | Sort-Object)
 [pscustomobject]@{placement=$group.Name;count=$values.Count;medianMilliseconds=$values[[int][Math]::Floor($values.Count/2)];minimum=$values[0];maximum=$values[-1]}
})
$summary | ConvertTo-Json | Set-Content "$OutputDirectory\summary.json"
