param([string]$Capability,[string]$ModelsDirectory,[string]$OutputDirectory)
$ErrorActionPreference='Stop'
$request=[Console]::In.ReadToEnd() | ConvertFrom-Json -AsHashtable
@{status='ok';capability=$Capability;request=$request;models=$ModelsDirectory;output=$OutputDirectory} | ConvertTo-Json -Depth 20 -Compress
