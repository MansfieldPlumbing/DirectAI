param([string]$DotNet='C:\Program Files\dotnet\dotnet.exe')
$ErrorActionPreference='Stop'
$env:DOTNET_CLI_HOME=Join-Path $env:USERPROFILE '.dotnet'
$env:NUGET_PACKAGES=Join-Path $env:USERPROFILE '.nuget\packages'
$env:NUGET_HTTP_CACHE_PATH=Join-Path $env:LOCALAPPDATA 'NuGet\v3-cache'
& $DotNet build "$PSScriptRoot\DirectAI.csproj" -c Release
if($LASTEXITCODE -ne 0){throw 'DirectAI build failed'}
$app=Join-Path $PSScriptRoot 'app'
New-Item -ItemType Directory -Path $app -Force | Out-Null
Copy-Item -Path "$PSScriptRoot\projects\bin\DirectAI.Host\Release\net11.0\*" -Destination $app -Recurse -Force
foreach($name in @('Diffusion','Erase','Segmentation','FaceSwap','Resize')){
 $destination=Join-Path $app "plugins\$name"
 New-Item -ItemType Directory -Path $destination -Force | Out-Null
 $source="$PSScriptRoot\projects\bin\DirectAI.$name\Release\net11.0"
 Copy-Item -Path "$source\*" -Destination $destination -Recurse -Force
 if(Test-Path "$source\runtimes\win-x64\native"){Copy-Item -Path "$source\runtimes\win-x64\native\*" -Destination $destination -Force}
}
$scriptPlugin=Join-Path $app 'plugins\scripts'
New-Item -ItemType Directory -Path $scriptPlugin -Force | Out-Null
Copy-Item -Path "$PSScriptRoot\src\ScriptExample\*" -Destination $scriptPlugin -Force
