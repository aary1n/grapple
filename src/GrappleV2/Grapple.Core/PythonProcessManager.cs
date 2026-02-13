using System;
using System.Diagnostics;
using System.IO;

namespace Grapple.Core
{
    /// <summary>
    /// Manages the Python GrappleDetector.py process as a sidecar.
    /// Handles spawning, output forwarding, and graceful/forced shutdown.
    /// </summary>
    public class PythonProcessManager : IDisposable
    {
        private Process? _process;
        private bool _disposed = false;

        // Configuration
        private readonly string _pythonPath;
        private readonly string _scriptPath;

        /// <summary>
        /// True if the Python process is currently running.
        /// </summary>
        public bool IsRunning => _process != null && !_process.HasExited;

        /// <summary>
        /// Creates a new Python process manager.
        /// </summary>
        /// <param name="pythonPath">Path to python.exe. Defaults to "py" (Windows py launcher).</param>
        /// <param name="scriptPath">Path to GrappleDetector.py (auto-detected if null).</param>
        public PythonProcessManager(string? pythonPath = null, string? scriptPath = null)
        {
            // Check environment variable for Python path override
            _pythonPath = pythonPath
                ?? Environment.GetEnvironmentVariable("GRAPPLE_PYTHON_PATH")
                ?? "py";

            _scriptPath = scriptPath ?? FindScriptPath();
        }

        /// <summary>
        /// Starts the Python detector process.
        /// </summary>
        /// <returns>True if process started successfully.</returns>
        public bool Start()
        {
            if (IsRunning)
            {
                Console.WriteLine("[Py] Process already running.");
                return true;
            }

            Console.WriteLine($"[*] Starting Python detector: {_pythonPath} -3.12 \"{Path.GetFileName(_scriptPath)}\"");

            try
            {
                var startInfo = new ProcessStartInfo
                {
                    FileName = _pythonPath,
                    Arguments = $"-3.12 \"{_scriptPath}\"",
                    CreateNoWindow = true,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    WorkingDirectory = Path.GetDirectoryName(_scriptPath) ?? Environment.CurrentDirectory
                };

                _process = new Process { StartInfo = startInfo };

                // Forward stdout only (Python warnings suppressed at source via env vars)
                _process.OutputDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                        Console.WriteLine($"[Py] {e.Data}");
                };

                // Forward stderr (critical for diagnosing Python failures)
                _process.ErrorDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                        Console.WriteLine($"[Py:ERR] {e.Data}");
                };

                if (!_process.Start())
                {
                    Console.WriteLine("[!] ERROR: Failed to start Python process.");
                    return false;
                }

                // Begin async reading of stdout/stderr
                _process.BeginOutputReadLine();
                _process.BeginErrorReadLine();

                Console.WriteLine($"[+] Python process started (PID: {_process.Id})");
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[!] Failed to start Python: {ex.Message}");
                Console.WriteLine("[!] Make sure Python 3.12 is installed and 'py -3.12' works.");
                return false;
            }
        }

        /// <summary>
        /// Stops the Python process (forced kill).
        /// </summary>
        /// <param name="timeoutMs">Milliseconds to wait before force kill.</param>
        public void Stop(int timeoutMs = 2000)
        {
            if (_process == null || _process.HasExited)
                return;

            Console.WriteLine("[*] Stopping Python process...");

            try
            {
                // Kill the process and entire process tree
                _process.Kill(entireProcessTree: true);

                if (!_process.WaitForExit(timeoutMs))
                {
                    Console.WriteLine("[!] Python process did not exit in time, forcing termination...");
                    try
                    {
                        _process.Kill();
                    }
                    catch { }
                }

                Console.WriteLine("[+] Python process stopped.");
            }
            catch (InvalidOperationException)
            {
                // Process already exited - this is fine
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[!] Error stopping Python: {ex.Message}");
            }
        }

        /// <summary>
        /// Auto-detect the script path based on where the binary is running.
        /// </summary>
        private static string FindScriptPath()
        {
            // Check environment variable first
            string? envPath = Environment.GetEnvironmentVariable("GRAPPLE_DETECTOR_PATH");
            if (!string.IsNullOrEmpty(envPath) && File.Exists(envPath))
                return envPath;

            // Start from the executable's directory
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;

            // Development mode: bin/Debug/net9.0/ → need to go up to find tools/
            // Path: ../../../../tools/GrappleDetector.py (from bin/Debug/net9.0/)
            string devPath = Path.Combine(baseDir, "..", "..", "..", "..", "tools", "GrappleDetector.py");
            devPath = Path.GetFullPath(devPath);
            if (File.Exists(devPath))
                return devPath;

            // Alternative dev path: might be running from Grapple.SmokeTests project root
            string altDevPath = Path.Combine(baseDir, "..", "tools", "GrappleDetector.py");
            altDevPath = Path.GetFullPath(altDevPath);
            if (File.Exists(altDevPath))
                return altDevPath;

            // Deployed mode: script might be copied alongside the exe
            string deployedPath = Path.Combine(baseDir, "tools", "GrappleDetector.py");
            if (File.Exists(deployedPath))
                return deployedPath;

            // Same directory as exe
            string sameDirPath = Path.Combine(baseDir, "GrappleDetector.py");
            if (File.Exists(sameDirPath))
                return sameDirPath;

            throw new FileNotFoundException(
                $"Could not find GrappleDetector.py. Searched:\n" +
                $"  - {devPath}\n" +
                $"  - {deployedPath}\n" +
                $"  - {sameDirPath}\n" +
                "Set GRAPPLE_DETECTOR_PATH environment variable to override.");
        }

        public void Dispose()
        {
            if (_disposed) return;

            Stop();
            _process?.Dispose();
            _disposed = true;

            GC.SuppressFinalize(this);
        }

        ~PythonProcessManager()
        {
            Dispose();
        }
    }
}

