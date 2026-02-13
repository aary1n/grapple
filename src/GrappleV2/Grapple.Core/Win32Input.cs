using System;
using System.Runtime.InteropServices;
using System.Threading;

namespace Grapple.Core
{
    /// <summary>
    /// P/Invoke wrapper for Windows cursor and mouse button control.
    /// Supports both legacy (SetCursorPos/mouse_event) and modern (SendInput) APIs.
    /// SendInput with VIRTUALDESK handles DPI scaling and multi-monitor natively.
    /// Auto-falls back to legacy APIs if SendInput fails (struct mismatch, UIPI, etc.).
    /// </summary>
    public static class Win32Input
    {
        #region Constants

        // GetSystemMetrics indices
        private const int SM_CXSCREEN = 0;          // Primary monitor width
        private const int SM_CYSCREEN = 1;          // Primary monitor height
        private const int SM_XVIRTUALSCREEN = 76;   // Virtual desktop left edge
        private const int SM_YVIRTUALSCREEN = 77;   // Virtual desktop top edge
        private const int SM_CXVIRTUALSCREEN = 78;  // Virtual desktop width
        private const int SM_CYVIRTUALSCREEN = 79;  // Virtual desktop height

        // mouse_event flags (legacy)
        private const uint MOUSEEVENTF_LEFTDOWN = 0x0002;
        private const uint MOUSEEVENTF_LEFTUP = 0x0004;

        // SendInput flags
        private const uint MOUSEEVENTF_MOVE = 0x0001;
        private const uint MOUSEEVENTF_ABSOLUTE = 0x8000;
        private const uint MOUSEEVENTF_VIRTUALDESK = 0x4000;
        private const uint INPUT_MOUSE = 0;

        // Virtual key codes
        public const int VK_F9 = 0x78;

        // SendInput fallback tracking
        private static int _sendInputFailCount;
        private static int _useLegacyFallback; // 0 = use SendInput, 1 = use legacy

        #endregion

        #region SendInput Structs

        [StructLayout(LayoutKind.Sequential)]
        private struct MOUSEINPUT
        {
            public int dx;
            public int dy;
            public uint mouseData;
            public uint dwFlags;
            public uint time;
            public UIntPtr dwExtraInfo;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct INPUT
        {
            public uint type;
            public MOUSEINPUT mi;
        }

        #endregion

        #region P/Invoke

        [DllImport("user32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool SetCursorPos(int x, int y);

        [DllImport("user32.dll")]
        private static extern int GetSystemMetrics(int nIndex);

        [DllImport("user32.dll")]
        private static extern void mouse_event(uint dwFlags, int dx, int dy, uint dwData, UIntPtr dwExtraInfo);

        [DllImport("user32.dll")]
        private static extern short GetAsyncKeyState(int vKey);

        [DllImport("user32.dll", SetLastError = true)]
        private static extern uint SendInput(uint nInputs, INPUT[] pInputs, int cbSize);

        #endregion

        #region Public Properties

        /// <summary>Primary screen width in pixels.</summary>
        public static int ScreenWidth { get; }

        /// <summary>Primary screen height in pixels.</summary>
        public static int ScreenHeight { get; }

        /// <summary>Virtual desktop width spanning all monitors.</summary>
        public static int VirtualScreenWidth { get; }

        /// <summary>Virtual desktop height spanning all monitors.</summary>
        public static int VirtualScreenHeight { get; }

        /// <summary>Left edge of virtual desktop (can be negative if monitor is left of primary).</summary>
        public static int VirtualScreenLeft { get; }

        /// <summary>Top edge of virtual desktop (can be negative if monitor is above primary).</summary>
        public static int VirtualScreenTop { get; }

        #endregion

        #region Static Constructor

        static Win32Input()
        {
            ScreenWidth = GetSystemMetrics(SM_CXSCREEN);
            ScreenHeight = GetSystemMetrics(SM_CYSCREEN);
            VirtualScreenWidth = GetSystemMetrics(SM_CXVIRTUALSCREEN);
            VirtualScreenHeight = GetSystemMetrics(SM_CYVIRTUALSCREEN);
            VirtualScreenLeft = GetSystemMetrics(SM_XVIRTUALSCREEN);
            VirtualScreenTop = GetSystemMetrics(SM_YVIRTUALSCREEN);

            // Validate struct sizes match native Windows API expectations
            int inputSize = Marshal.SizeOf<INPUT>();
            int mouseInputSize = Marshal.SizeOf<MOUSEINPUT>();
            bool isX64 = IntPtr.Size == 8;
            int expectedInput = isX64 ? 40 : 28;
            int expectedMouse = isX64 ? 32 : 24;

            GrappleLogger.Info("Input", $"SendInput structs: INPUT={inputSize} (expect {expectedInput}), MOUSEINPUT={mouseInputSize} (expect {expectedMouse}), arch={(isX64 ? "x64" : "x86")}");

            if (inputSize != expectedInput || mouseInputSize != expectedMouse)
            {
                GrappleLogger.Warning("Input", $"Struct size MISMATCH — SendInput will likely fail. Forcing legacy fallback.");
                Volatile.Write(ref _useLegacyFallback, 1);
            }
        }

        #endregion

        #region Legacy Methods (Primary Monitor Only)

        /// <summary>
        /// Moves the cursor to an absolute screen position (primary monitor only).
        /// </summary>
        public static bool MoveMouse(int x, int y)
        {
            return SetCursorPos(x, y);
        }

        /// <summary>
        /// Sends a left mouse button down event (legacy mouse_event API).
        /// </summary>
        public static void LeftDown()
        {
            mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, UIntPtr.Zero);
        }

        /// <summary>
        /// Sends a left mouse button up event (legacy mouse_event API).
        /// </summary>
        public static void LeftUp()
        {
            mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, UIntPtr.Zero);
        }

        #endregion

        #region SendInput Methods (DPI-Aware, Multi-Monitor)

        /// <summary>
        /// Moves the cursor using SendInput with VIRTUALDESK flag.
        /// Handles DPI scaling and multi-monitor natively.
        /// Auto-falls back to SetCursorPos if SendInput fails.
        /// </summary>
        /// <param name="normalizedX">X position in 0.0-1.0 range across virtual desktop</param>
        /// <param name="normalizedY">Y position in 0.0-1.0 range across virtual desktop</param>
        public static void MoveMouseVirtual(double normalizedX, double normalizedY)
        {
            if (Volatile.Read(ref _useLegacyFallback) == 0)
            {
                int absX = (int)(normalizedX * 65535.0);
                int absY = (int)(normalizedY * 65535.0);

                var input = new INPUT[1];
                input[0].type = INPUT_MOUSE;
                input[0].mi.dx = absX;
                input[0].mi.dy = absY;
                input[0].mi.mouseData = 0;
                input[0].mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK;
                input[0].mi.time = 0;
                input[0].mi.dwExtraInfo = UIntPtr.Zero;

                uint result = SendInput(1, input, Marshal.SizeOf<INPUT>());
                if (result == 1) return; // Success

                // SendInput failed
                int failCount = Interlocked.Increment(ref _sendInputFailCount);
                int error = Marshal.GetLastWin32Error();

                if (failCount <= 3)
                {
                    GrappleLogger.Warning("Input", $"SendInput MOVE failed (error={error}, INPUT size={Marshal.SizeOf<INPUT>()}, fail #{failCount})");
                }

                if (failCount >= 3)
                {
                    GrappleLogger.Warning("Input", "SendInput consistently failing — switching to SetCursorPos/mouse_event fallback permanently");
                    Volatile.Write(ref _useLegacyFallback, 1);
                }
            }

            // Legacy fallback: convert normalized coords to virtual desktop pixels
            int pixelX = VirtualScreenLeft + (int)(normalizedX * VirtualScreenWidth);
            int pixelY = VirtualScreenTop + (int)(normalizedY * VirtualScreenHeight);
            SetCursorPos(pixelX, pixelY);
        }

        /// <summary>
        /// Sends a left mouse button down event via SendInput.
        /// Falls back to mouse_event if SendInput is unavailable.
        /// </summary>
        public static void LeftDownSendInput()
        {
            if (Volatile.Read(ref _useLegacyFallback) == 1)
            {
                mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, UIntPtr.Zero);
                return;
            }

            var input = new INPUT[1];
            input[0].type = INPUT_MOUSE;
            input[0].mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
            input[0].mi.dwExtraInfo = UIntPtr.Zero;

            uint result = SendInput(1, input, Marshal.SizeOf<INPUT>());
            if (result == 0)
            {
                mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, UIntPtr.Zero);
            }
        }

        /// <summary>
        /// Sends a left mouse button up event via SendInput.
        /// Falls back to mouse_event if SendInput is unavailable.
        /// </summary>
        public static void LeftUpSendInput()
        {
            if (Volatile.Read(ref _useLegacyFallback) == 1)
            {
                mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, UIntPtr.Zero);
                return;
            }

            var input = new INPUT[1];
            input[0].type = INPUT_MOUSE;
            input[0].mi.dwFlags = MOUSEEVENTF_LEFTUP;
            input[0].mi.dwExtraInfo = UIntPtr.Zero;

            uint result = SendInput(1, input, Marshal.SizeOf<INPUT>());
            if (result == 0)
            {
                mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, UIntPtr.Zero);
            }
        }

        #endregion

        #region Utility Methods

        /// <summary>
        /// Checks if a key is currently pressed.
        /// Uses GetAsyncKeyState which works even when the app doesn't have focus.
        /// </summary>
        public static bool IsKeyDown(int vKey)
        {
            return (GetAsyncKeyState(vKey) & 0x8000) != 0;
        }

        /// <summary>
        /// Computes the absolute SendInput coordinate (0-65535) for a normalized value.
        /// Useful for unit testing coordinate mapping.
        /// </summary>
        public static int NormalizedToAbsolute(double normalized)
        {
            return (int)(Math.Clamp(normalized, 0.0, 1.0) * 65535.0);
        }

        #endregion
    }
}

