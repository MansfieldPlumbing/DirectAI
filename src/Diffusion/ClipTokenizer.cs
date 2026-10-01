using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace DirectAI;

public class ClipTokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<(string, string), int> _bpeRanks;
    private readonly Regex _tokenRegex;
    private readonly Dictionary<byte, char> _byteEncoder;

    public const int MaxTokenLength = 77;
    public const int StartTokenId = 49406; // <|startoftext|>
    public const int EndTokenId = 49407;   // <|endoftext|>

    public ClipTokenizer(string tokenizerDir)
    {
        string vocabPath = Path.Combine(tokenizerDir, "vocab.json");
        string mergesPath = Path.Combine(tokenizerDir, "merges.txt");

        if (!File.Exists(vocabPath))
            throw new FileNotFoundException($"vocab.json not found in {tokenizerDir}");
        if (!File.Exists(mergesPath))
            throw new FileNotFoundException($"merges.txt not found in {tokenizerDir}");

        string vocabJson = File.ReadAllText(vocabPath);
        _vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(vocabJson);

        _byteEncoder = BytesToUnicode();
        _bpeRanks = LoadMerges(mergesPath);

        _tokenRegex = new Regex(@"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+", RegexOptions.Compiled | RegexOptions.IgnoreCase);
    }

    private static Dictionary<byte, char> BytesToUnicode()
    {
        var bs = new List<int>();
        for (int b = '!'; b <= '~'; b++) bs.Add(b);
        for (int b = '¡'; b <= '¬'; b++) bs.Add(b);
        for (int b = '®'; b <= 'ÿ'; b++) bs.Add(b);

        var cs = new List<int>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!bs.Contains(b))
            {
                bs.Add(b);
                cs.Add(256 + n);
                n++;
            }
        }

        var map = new Dictionary<byte, char>();
        for (int i = 0; i < bs.Count; i++)
        {
            map[(byte)bs[i]] = (char)cs[i];
        }
        return map;
    }

    private static Dictionary<(string, string), int> LoadMerges(string mergesPath)
    {
        var ranks = new Dictionary<(string, string), int>();
        int rank = 0;
        foreach (var line in File.ReadLines(mergesPath))
        {
            var trimmed = line.Trim();
            if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith("#"))
                continue;

            var parts = trimmed.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 2)
            {
                ranks[(parts[0], parts[1])] = rank++;
            }
        }
        return ranks;
    }

    public int[] Tokenize(string text)
    {
        var tokens = new List<int> { StartTokenId };
        if (!string.IsNullOrWhiteSpace(text))
        {
            text = text.ToLowerInvariant();
            var matches = _tokenRegex.Matches(text);

            foreach (Match match in matches)
            {
                var tokenBytes = Encoding.UTF8.GetBytes(match.Value);
                var sb = new StringBuilder();
                foreach (var b in tokenBytes)
                {
                    if (_byteEncoder.TryGetValue(b, out char c))
                        sb.Append(c);
                }

                var bpeTokens = Bpe(sb.ToString());
                foreach (var bpeToken in bpeTokens)
                {
                    if (_vocab.TryGetValue(bpeToken, out int id))
                    {
                        tokens.Add(id);
                        if (tokens.Count >= MaxTokenLength - 1)
                            break;
                    }
                }
                if (tokens.Count >= MaxTokenLength - 1)
                    break;
            }
        }

        tokens.Add(EndTokenId);

        // Pad to 77 tokens with EndTokenId (or 0)
        var result = new int[MaxTokenLength];
        for (int i = 0; i < MaxTokenLength; i++)
        {
            result[i] = i < tokens.Count ? tokens[i] : EndTokenId;
        }

        return result;
    }

    private List<string> Bpe(string token)
    {
        if (token.Length <= 1)
            return new List<string> { token + "</w>" };

        var word = new List<string>();
        for (int i = 0; i < token.Length - 1; i++)
            word.Add(token[i].ToString());
        word.Add(token[^1] + "</w>");

        var pairs = GetPairs(word);
        if (pairs.Count == 0)
            return new List<string> { token + "</w>" };

        while (true)
        {
            (string, string) minPair = default;
            int minRank = int.MaxValue;
            bool found = false;

            foreach (var pair in pairs)
            {
                if (_bpeRanks.TryGetValue(pair, out int r) && r < minRank)
                {
                    minRank = r;
                    minPair = pair;
                    found = true;
                }
            }

            if (!found) break;

            var newWord = new List<string>();
            int i = 0;
            while (i < word.Count)
            {
                if (i < word.Count - 1 && word[i] == minPair.Item1 && word[i + 1] == minPair.Item2)
                {
                    newWord.Add(minPair.Item1 + minPair.Item2);
                    i += 2;
                }
                else
                {
                    newWord.Add(word[i]);
                    i++;
                }
            }

            word = newWord;
            if (word.Count == 1) break;
            pairs = GetPairs(word);
        }

        return word;
    }

    private static List<(string, string)> GetPairs(List<string> word)
    {
        var pairs = new List<(string, string)>();
        for (int i = 0; i < word.Count - 1; i++)
        {
            pairs.Add((word[i], word[i + 1]));
        }
        return pairs;
    }
}
