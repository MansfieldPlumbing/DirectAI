using System.Text.RegularExpressions;

namespace DirectAI.ModelConversion;

internal sealed record WeightBinding(string[] Sources, string Transform, int Heads = 1);
internal static class CheckpointMappings
{
    public static WeightBinding Resolve(string stage, OnnxTensor tensor, OnnxInspection graph, bool xl, IReadOnlyDictionary<string, SourceTensor> source)
    {
        var consumers = graph.Nodes.Where(n => n.Inputs.Contains(tensor.Name)).ToArray();
        if (consumers.Length == 0) throw new InvalidDataException($"Unused floating initializer {stage}:{tensor.Name}.");
        string name = tensor.Name;
        bool transpose = false;
        if (Regex.IsMatch(name, "^MatMul_(QKV|KV)_"))
        {
            var reshape = graph.Nodes.Single(n => n.Operation == "Reshape" && consumers.Any(c => c.Outputs.Intersect(n.Inputs).Any()));
            var attention = graph.Nodes.Single(n => n.Operation == "MultiHeadAttention" && n.Inputs.Intersect(reshape.Outputs).Any());
            string scope = Scope(attention.Outputs[0]);
            int heads = checked((int)attention.IntegerAttributes["num_heads"]);
            string[] suffixes = name.StartsWith("MatMul_QKV_") ? ["to_q.weight", "to_k.weight", "to_v.weight"] : ["to_k.weight", "to_v.weight"];
            return new(suffixes.Select(s => Unet(scope + "." + s, source)).ToArray(), "packed-head-projections", heads);
        }
        if (name.StartsWith("GroupNorm_"))
        {
            var norm = consumers.Single(n => n.Operation == "GroupNorm");
            string scope = Scope(norm.Outputs[0]);
            if (stage == "unet")
            {
                if (scope == "act") scope = "conv_norm_out";
                else if (Regex.IsMatch(scope, @"\.act(_\d+)?$"))
                    scope = Regex.Replace(scope, @"\.act(_\d+)?$", scope.EndsWith(".act") ? ".norm1" : ".norm2");
            }
            else
            {
                scope = Regex.Replace(scope, @"\.nonlinearity(_\d+)?$", scope.EndsWith(".nonlinearity") ? ".norm1" : ".norm2");
                scope = scope.Replace(".conv_act", ".conv_norm_out");
            }
            name = scope + (name.EndsWith("_gamma") ? ".weight" : ".bias");
        }
        else if (name.StartsWith("onnx::"))
        {
            var node = consumers.First();
            string suffix = node.Operation switch
            {
                "MatMul" or "Conv" => ".weight",
                "Gemm" => Array.IndexOf(node.Inputs, tensor.Name) == 1 ? ".weight" : ".bias",
                "Mul" => ".weight",
                "Add" => ".bias",
                _ => throw new InvalidDataException($"Unresolved initializer {stage}:{name}, consumer {node.Name}/{node.Operation}.")
            };
            name = Scope(node.Name) + suffix;
        }
        var matrix = consumers.FirstOrDefault(n => n.Operation is "MatMul" or "Gemm" && Array.IndexOf(n.Inputs, tensor.Name) == 1);
        if (matrix != null) transpose = matrix.Operation == "MatMul" || matrix.IntegerAttributes.GetValueOrDefault("transB") == 0;
        string key = stage switch
        {
            "unet" => Unet(name, source),
            "text_encoder" => (xl ? "conditioner.embedders.0.transformer." : "cond_stage_model.transformer.") + name,
            "vae_decoder" or "vae_encoder" => Vae(name),
            _ => throw new NotSupportedException($"No validated binding recipe for stage {stage}.")
        };
        return new([key], transpose ? "transpose" : "identity-or-reshape");
    }
    private static string Scope(string name)
    {
        var pieces = name.Trim('/').Split('/');
        if (pieces.Length < 2) throw new InvalidDataException($"No semantic scope retained in {name}.");
        return string.Join('.', pieces.Take(pieces.Length - 1));
    }
    private static string Unet(string name, IReadOnlyDictionary<string, SourceTensor> source)
    {
        bool residualBlock = name.Contains(".resnets.", StringComparison.Ordinal);
        name = name switch
        {
            "conv_in.weight" => "input_blocks.0.0.weight",
            "conv_in.bias" => "input_blocks.0.0.bias",
            _ => name
        };
        name = name.Replace("time_embedding.linear_1.", "time_embed.0.").Replace("time_embedding.linear_2.", "time_embed.2.");
        name = name.Replace("add_embedding.linear_1.", "label_emb.0.0.").Replace("add_embedding.linear_2.", "label_emb.0.2.");
        name = name.Replace("conv_norm_out.", "out.0.").Replace("conv_out.", "out.2.");
        name = Regex.Replace(name, @"^down_blocks\.(\d+)\.(resnets|attentions)\.(\d+)\.", m => $"input_blocks.{1 + 3 * int.Parse(m.Groups[1].Value) + int.Parse(m.Groups[3].Value)}.{(m.Groups[2].Value == "resnets" ? 0 : 1)}.");
        name = Regex.Replace(name, @"^down_blocks\.(\d+)\.downsamplers\.0\.conv\.", m => $"input_blocks.{3 * (int.Parse(m.Groups[1].Value) + 1)}.0.op.");
        name = name.Replace("mid_block.resnets.0.", "middle_block.0.").Replace("mid_block.attentions.0.", "middle_block.1.").Replace("mid_block.resnets.1.", "middle_block.2.");
        name = Regex.Replace(name, @"^up_blocks\.(\d+)\.(resnets|attentions)\.(\d+)\.", m => $"output_blocks.{3 * int.Parse(m.Groups[1].Value) + int.Parse(m.Groups[3].Value)}.{(m.Groups[2].Value == "resnets" ? 0 : 1)}.");
        name = Regex.Replace(name, @"^up_blocks\.(\d+)\.upsamplers\.0\.conv\.", m =>
        {
            int block = 3 * int.Parse(m.Groups[1].Value) + 2;
            return source.ContainsKey($"model.diffusion_model.output_blocks.{block}.1.conv.weight") ? $"output_blocks.{block}.1.conv." : $"output_blocks.{block}.2.conv.";
        });
        if (residualBlock)
            name = name.Replace(".norm1.", ".in_layers.0.").Replace(".conv1.", ".in_layers.2.").Replace(".norm2.", ".out_layers.0.").Replace(".conv2.", ".out_layers.3.").Replace(".time_emb_proj.", ".emb_layers.1.").Replace(".conv_shortcut.", ".skip_connection.");
        return "model.diffusion_model." + name;
    }
    private static string Vae(string name)
    {
        name = name.Replace(".conv_norm_out.", ".norm_out.");
        name = Regex.Replace(name, @"^(encoder|decoder)\.mid_block\.resnets\.(\d+)\.", m => $"{m.Groups[1].Value}.mid.block_{int.Parse(m.Groups[2].Value) + 1}.");
        name = Regex.Replace(name, @"^(encoder|decoder)\.mid_block\.attentions\.0\.", "$1.mid.attn_1.");
        name = name.Replace(".group_norm.", ".norm.").Replace(".to_q.", ".q.").Replace(".to_k.", ".k.").Replace(".to_v.", ".v.").Replace(".to_out.0.", ".proj_out.");
        name = Regex.Replace(name, @"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.", m => $"decoder.up.{3 - int.Parse(m.Groups[1].Value)}.block.{m.Groups[2].Value}.");
        name = Regex.Replace(name, @"^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.conv\.", m => $"decoder.up.{3 - int.Parse(m.Groups[1].Value)}.upsample.conv.");
        name = Regex.Replace(name, @"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.", "encoder.down.$1.block.$2.");
        name = Regex.Replace(name, @"^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.conv\.", "encoder.down.$1.downsample.conv.");
        name = name.Replace(".conv_shortcut.", ".nin_shortcut.");
        return "first_stage_model." + name;
    }
}
