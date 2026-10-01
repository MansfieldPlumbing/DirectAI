$script:Root=Split-Path $PSScriptRoot

function Get-DirectAICapability {
 [CmdletBinding()]param([string]$PluginDirectory=(Join-Path $script:Root 'app\plugins'))
 Get-ChildItem -LiteralPath $PluginDirectory -Filter plugin.json -Recurse -File | ForEach-Object {
  $manifest=Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
  foreach($capability in $manifest.capabilities){[pscustomobject]@{Plugin=$manifest.id;Id=$capability.id;Description=$capability.description;Manifest=$_.FullName}}
 }
}

function New-DirectAIRuntime {
 [CmdletBinding()]param([string]$ModelsDirectory='C:\Models\DirectAI',[string]$OutputDirectory=(Join-Path $script:Root 'output'),[string]$PluginDirectory=(Join-Path $script:Root 'app\plugins'))
 foreach($assembly in @('DirectAI.Core.dll','DirectAI.Routing.dll')){
  [Reflection.Assembly]::LoadFrom((Join-Path "$script:Root\app" $assembly)) | Out-Null
 }
 $context=[DirectAI.PluginContext]::new($ModelsDirectory,$OutputDirectory,'C:\bin\pwsh\pwsh.exe')
 $registry=[DirectAI.PluginRegistry]::new($PluginDirectory,$context)
 [pscustomobject]@{Registry=$registry;Router=[DirectAI.RequestRouter]::new($registry)}
}

function Invoke-DirectAI {
 [CmdletBinding()]param([Parameter(Mandatory)]$Runtime,[Parameter(Mandatory)][string]$Capability,[hashtable]$Request=@{})
 $document=[Text.Json.JsonDocument]::Parse(($Request | ConvertTo-Json -Depth 30 -Compress))
 try{$Runtime.Router.InvokeAsync($Capability,$document.RootElement,[Threading.CancellationToken]::None).GetAwaiter().GetResult()}
 finally{$document.Dispose()}
}

function Get-DirectAITelemetry {
 [CmdletBinding()]param([Parameter(Mandatory)]$Runtime)
 $Runtime.Registry.Telemetry()
}

function Remove-DirectAIRuntime {
 [CmdletBinding()]param([Parameter(Mandatory)]$Runtime)
 $Runtime.Registry.Dispose()
}
Export-ModuleMember -Function Get-DirectAICapability,New-DirectAIRuntime,Invoke-DirectAI,Get-DirectAITelemetry,Remove-DirectAIRuntime
