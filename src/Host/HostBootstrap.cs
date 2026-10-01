namespace DirectAI;
public static class HostBootstrap
{
    public static async Task RunAsync(string[] args)
    {
        string Get(string key, string fallback) { int index = Array.IndexOf(args, key); return index >= 0 && index + 1 < args.Length ? args[index + 1] : fallback; }
        string root = Get("--root", Environment.GetEnvironmentVariable("DIRECTAI_ROOT") ?? Environment.CurrentDirectory);
        using var registry = new PluginRegistry(Get("--plugins", Path.Combine(root, "app", "plugins")), new PluginContext(Get("--models", @"C:\Models\DirectAI"), Get("--output-dir", Path.Combine(root, "output"))));
        var router = new RequestRouter(registry);
        if (args.FirstOrDefault() == "server")
        {
            using var cancellation = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) => { e.Cancel = true; cancellation.Cancel(); };
            using var http = new HttpTransport(router.InvokeAsync, int.Parse(Get("--port", "5050")), Path.Combine(AppContext.BaseDirectory, "studio.html"));
            var pipes = new NamedPipeTransport(router.InvokeAsync);
            Console.Error.WriteLine("DirectAI HTTP and named-pipe adapters started.");
            await Task.WhenAll(http.RunAsync(cancellation.Token), pipes.RunAsync(cancellation.Token));
        }
        else
        {
            var stdout = Console.Out; Console.SetOut(Console.Error);
            try { Environment.ExitCode = await new CliTransport(router.InvokeAsync).RunAsync(args, stdout, Console.Error); }
            finally { Console.SetOut(stdout); }
        }
    }
}
