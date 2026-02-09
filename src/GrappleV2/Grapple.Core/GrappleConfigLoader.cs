using System;
using System.IO;
using System.Text.Json;

namespace Grapple.Core
{
    /// <summary>
    /// Loads GrappleConfig from grapple_config.json.
    /// Falls back to defaults if file not found.
    /// Called ONCE at startup. Result is treated as immutable for the process lifetime.
    /// </summary>
    public static class GrappleConfigLoader
    {
        private static readonly JsonSerializerOptions JsonOptions = new()
        {
            PropertyNameCaseInsensitive = true,
            ReadCommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true
        };

        /// <summary>
        /// Loads config from the given path, or searches for grapple_config.json.
        /// Returns defaults if file not found.
        /// </summary>
        public static GrappleConfig Load(string? configPath = null)
        {
            configPath ??= FindConfigFile();

            if (configPath == null || !File.Exists(configPath))
            {
                Console.WriteLine("[Config] No grapple_config.json found. Using defaults.");
                return new GrappleConfig();
            }

            try
            {
                string json = File.ReadAllText(configPath);
                var config = JsonSerializer.Deserialize<GrappleConfig>(json, JsonOptions)
                             ?? new GrappleConfig();
                Console.WriteLine($"[Config] Loaded from {configPath}");
                return config;
            }
            catch (JsonException ex)
            {
                Console.WriteLine($"[Config] WARNING: Failed to parse {configPath}: {ex.Message}");
                Console.WriteLine("[Config] Using defaults.");
                return new GrappleConfig();
            }
        }

        /// <summary>
        /// Searches for grapple_config.json by walking up from the executable's directory.
        /// </summary>
        private static string? FindConfigFile()
        {
            string dir = AppDomain.CurrentDomain.BaseDirectory;
            for (int i = 0; i < 8; i++)
            {
                string candidate = Path.Combine(dir, "grapple_config.json");
                if (File.Exists(candidate))
                    return candidate;

                string? parent = Directory.GetParent(dir)?.FullName;
                if (parent == null)
                    break;
                dir = parent;
            }
            return null;
        }
    }
}
