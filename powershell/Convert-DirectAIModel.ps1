param(
 [Parameter(Mandatory)][string]$Checkpoint,
 [Parameter(Mandatory)][string]$OutputDirectory,
 [string]$TemplateDirectory='C:\Models\DirectAI\CyberRealistic-LCM-amuse',
 [ValidateSet('sd15')][string]$Architecture='sd15',
 [switch]$Build
)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$assembly=Join-Path $root 'projects\bin\DirectAI.ModelConversion\Release\net11.0\DirectAI.ModelConversion.dll'
if($Build -or -not(Test-Path $assembly)){
 & 'C:\Program Files\dotnet\dotnet.exe' build (Join-Path $root 'projects\DirectAI.ModelConversion.csproj') -c Release
 if($LASTEXITCODE -ne 0){throw 'Model conversion build failed.'}
}
[Reflection.Assembly]::LoadFrom($assembly) | Out-Null
[DirectAI.ModelConversion.CheckpointConverter]::Convert($Checkpoint,$TemplateDirectory,$OutputDirectory,$Architecture)
