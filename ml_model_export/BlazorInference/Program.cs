using System.Runtime.InteropServices;
using BlazorInference.Components;
using BlazorInference.Services;
using Microsoft.ML.OnnxRuntime;

// On Linux the managed wrapper DllImports "onnxruntime.dll" (Windows name).
// .NET can't resolve that to libonnxruntime.so automatically, so we do it here
// before any OnnxRuntime type is touched.
if (!OperatingSystem.IsWindows())
{
    NativeLibrary.SetDllImportResolver(
        typeof(InferenceSession).Assembly,
        (libraryName, assembly, searchPath) =>
        {
            if (libraryName.Equals("onnxruntime.dll", StringComparison.OrdinalIgnoreCase)
             || libraryName.Equals("onnxruntime",     StringComparison.OrdinalIgnoreCase))
            {
                // Look next to the managed DLL first (bin/…/runtimes/linux-x64/native/)
                var nativeDir = Path.Combine(AppContext.BaseDirectory, "runtimes", "linux-x64", "native");
                var soPath    = Path.Combine(nativeDir, "libonnxruntime.so");
                if (File.Exists(soPath) && NativeLibrary.TryLoad(soPath, out var handle))
                    return handle;

                // Fallback: let the OS find it on LD_LIBRARY_PATH / system paths
                if (NativeLibrary.TryLoad("libonnxruntime.so", assembly, searchPath, out handle))
                    return handle;
            }
            return IntPtr.Zero;
        });
}

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

// Register ONNX prediction service (singleton — loads model once at startup)
builder.Services.AddSingleton<OnnxPredictionService>();

var app = builder.Build();

// Eagerly load the model at startup so errors surface immediately
app.Services.GetRequiredService<OnnxPredictionService>();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    app.UseHsts();
}

app.UseHttpsRedirection();

app.UseStaticFiles();
app.UseAntiforgery();

app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
