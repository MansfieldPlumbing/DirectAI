param(
 [string]$Checkpoint='N:\models\checkpoints\cyberrealisticLCM_cyberrealistic42.safetensors',
 [string]$Onnx='C:\Models\DirectAI\CyberRealistic-LCM-amuse\unet\model.onnx',
 [switch]$AllUnetWeights,
 [string]$SourcePrefix='model.diffusion_model.'
)
$ErrorActionPreference='Stop'
function Read-Varint([IO.BinaryReader]$Reader){
 [long]$value=0
 for($shift=0;$shift -lt 63;$shift+=7){
  $next=$Reader.ReadByte()
  $value=$value -bor ([long]($next -band 127) -shl $shift)
  if(($next -band 128) -eq 0){return $value}
 }
 throw 'Oversized protobuf varint'
}
function Skip-Field([IO.BinaryReader]$Reader,[int]$Wire){
 switch($Wire){
  0{Read-Varint $Reader | Out-Null}
  1{$Reader.BaseStream.Seek(8,[IO.SeekOrigin]::Current) | Out-Null}
  2{$length=Read-Varint $Reader; $Reader.BaseStream.Seek($length,[IO.SeekOrigin]::Current) | Out-Null}
  5{$Reader.BaseStream.Seek(4,[IO.SeekOrigin]::Current) | Out-Null}
  default{throw "Unsupported protobuf wire type $Wire"}
 }
}
$checkpointReader=[IO.BinaryReader]::new([IO.File]::OpenRead($Checkpoint))
$onnxReader=[IO.BinaryReader]::new([IO.File]::OpenRead($Onnx))
try{
 $headerLength=$checkpointReader.ReadInt64()
 $header=[Text.Encoding]::UTF8.GetString($checkpointReader.ReadBytes([int]$headerLength)) | ConvertFrom-Json -AsHashtable
 $hashes=@{}
 $names=if($AllUnetWeights){@($header.Keys | Where-Object {$_.StartsWith($SourcePrefix,[StringComparison]::Ordinal)})}else{@('model.diffusion_model.input_blocks.0.0.weight','model.diffusion_model.out.2.weight')}
 foreach($name in $names){
  $tensor=$header[$name]
  if(-not $tensor){throw "Checkpoint tensor missing: $name"}
  $checkpointReader.BaseStream.Position=8+$headerLength+$tensor.data_offsets[0]
  $bytes=$checkpointReader.ReadBytes([int]($tensor.data_offsets[1]-$tensor.data_offsets[0]))
  $hash=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes))
  $hashes[$hash]=@{checkpointName=$name;dtype=$tensor.dtype;byteCount=$bytes.Length;shape=$tensor.shape}
 }
 while($onnxReader.BaseStream.Position -lt $onnxReader.BaseStream.Length){
  $tag=Read-Varint $onnxReader
  if(($tag -shr 3) -ne 7){Skip-Field $onnxReader ($tag -band 7);continue}
  $graphLength=Read-Varint $onnxReader
  $graphEnd=$onnxReader.BaseStream.Position+$graphLength
  while($onnxReader.BaseStream.Position -lt $graphEnd){
   $tag=Read-Varint $onnxReader
   if(($tag -shr 3) -ne 5){Skip-Field $onnxReader ($tag -band 7);continue}
   $tensorLength=Read-Varint $onnxReader
   $tensorEnd=$onnxReader.BaseStream.Position+$tensorLength
   $name='';$dtype=0;$rawStart=0L;$rawLength=0L
   while($onnxReader.BaseStream.Position -lt $tensorEnd){
    $tag=Read-Varint $onnxReader
    switch($tag -shr 3){
     2{$dtype=Read-Varint $onnxReader}
     8{$length=Read-Varint $onnxReader;$name=[Text.Encoding]::UTF8.GetString($onnxReader.ReadBytes([int]$length))}
     9{$rawLength=Read-Varint $onnxReader;$rawStart=$onnxReader.BaseStream.Position;$onnxReader.BaseStream.Position+=$rawLength}
     default{Skip-Field $onnxReader ($tag -band 7)}
    }
   }
   if(($AllUnetWeights -and $rawLength -ge 256 -and $dtype -eq 10) -or ($rawLength -eq 23040 -and $dtype -eq 10)){
    $onnxReader.BaseStream.Position=$rawStart
    $hash=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($onnxReader.ReadBytes([int]$rawLength)))
    [pscustomobject]@{onnxName=$name;sha256=$hash;checkpointMatch=$hashes[$hash]}
   }
   $onnxReader.BaseStream.Position=$tensorEnd
  }
 }
}finally{$checkpointReader.Dispose();$onnxReader.Dispose()}
