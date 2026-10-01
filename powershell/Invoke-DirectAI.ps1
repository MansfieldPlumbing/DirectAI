param([string]$Capability='capabilities',[string]$RequestPath)
$ErrorActionPreference='Stop'
Import-Module "$PSScriptRoot\DirectAI.psm1" -Force
if($Capability -eq 'capabilities'){Get-DirectAICapability;exit}
$runtime=New-DirectAIRuntime
try{
 $request=if($RequestPath){Get-Content -LiteralPath $RequestPath -Raw | ConvertFrom-Json -AsHashtable}else{@{}}
 Invoke-DirectAI -Runtime $runtime -Capability $Capability -Request $request
}finally{Remove-DirectAIRuntime $runtime}
