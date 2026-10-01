param([Parameter(ValueFromRemainingArguments=$true)][string[]]$ApplicationArguments)
$ErrorActionPreference='Stop'
# Process-local paths prevent inherited, locked-drive settings from breaking .NET.
$env:DOTNET_CLI_HOME=Join-Path $env:USERPROFILE '.dotnet'
$env:NUGET_PACKAGES=Join-Path $env:USERPROFILE '.nuget\packages'
$env:NUGET_HTTP_CACHE_PATH=Join-Path $env:LOCALAPPDATA 'NuGet\v3-cache'
$env:NUGET_PLUGINS_CACHE_PATH=Join-Path $env:LOCALAPPDATA 'NuGet\plugins-cache'
$env:DIRECTAI_OUTPUT_DIR=Join-Path $PSScriptRoot 'output'
foreach($taskDirectory in @($env:DOTNET_CLI_HOME,$env:NUGET_PACKAGES,$env:NUGET_HTTP_CACHE_PATH,$env:NUGET_PLUGINS_CACHE_PATH,$env:DIRECTAI_OUTPUT_DIR)){
 New-Item -ItemType Directory -Path $taskDirectory -Force | Out-Null
}
Push-Location $PSScriptRoot
try{& "$PSScriptRoot\app\DirectAI.Host.exe" @ApplicationArguments; exit $LASTEXITCODE}
finally{Pop-Location}
