using System;
using System.Runtime.InteropServices;

namespace Grapple.Core
{
    /// <summary>
    /// P/Invoke wrapper for Windows SendInput API.
    /// Provides absolute cursor positioning for hand-controlled mouse input.
    /// </summary>
    public static class Win32Input
    {
        #region Constants

        private const uint INPUT_MOUSE = 0;
        private const uint MOUSEEVENTF_MOVE = 0x0001;
        private const uint MOUSEEVENTF_ABSOLUTE = 0x8000;
        private const uint MOUSEEVENTF_VIRTUALDESK = 0x4000;

        private const int SM_CXSCREEN = 0;   // Primary monitor width
        private const int SM_CYSCREEN = 1;   // Primary monitor height

        #endregion

        #region Win32 Structs

        [StructLayout(LayoutKind.Sequential)]
        private struct MOUSEINPUT
        {
            public int dx;
            public int dy;
            public uint mouseData;
            public uint dwFlags;
            public uint time;
            public IntPtr dwExtraInfo;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct KEYBDINPUT
        {
            public ushort wVk;
            public ushort wScan;
            public uint dwFlags;
            public uint time;
            public IntPtr dwExtraInfo;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct HARDWAREINPUT
        {
            public uint uMsg;
            public ushort wParamL;
            public ushort wParamH;
        }

        [StructLayout(LayoutKind.Explicit)]
        private struct INPUT
        {
            [FieldOffset(0)]
            public uint type;

            [FieldOffset(4)]
            public MOUSEINPUT mi;

            [FieldOffset(4)]
            public KEYBDINPUT ki;

            [FieldOffset(4)]
            public HARDWAREINPUT hi;
        }

        #endregion

        #region P/Invoke

        [DllImport("user32.dll", SetLastError = true)]
        private static extern uint SendInput(uint nInputs, INPUT[] pInputs, int cbSize);

        [DllImport("user32.dll")]
        private static extern int GetSystemMetrics(int nIndex);

        #endregion

        #region Public Properties

        /// <summary>
        /// Primary screen width in pixels.
        /// </summary>
        public static int ScreenWidth { get; }

        /// <summary>
        /// Primary screen height in pixels.
        /// </summary>
        public static int ScreenHeight { get; }

        #endregion

        #region Static Constructor

        static Win32Input()
        {
            ScreenWidth = GetSystemMetrics(SM_CXSCREEN);
            ScreenHeight = GetSystemMetrics(SM_CYSCREEN);
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Moves the cursor to an absolute screen position.
        /// </summary>
        /// <param name="x">Screen X coordinate (0 to ScreenWidth-1)</param>
        /// <param name="y">Screen Y coordinate (0 to ScreenHeight-1)</param>
        public static void MoveMouse(int x, int y)
        {
            // Windows absolute coordinates are normalized to 0-65535 range
            // Use 65536L (not 65535) to avoid off-by-one at screen edges
            int absX = (int)((x * 65536L) / ScreenWidth);
            int absY = (int)((y * 65536L) / ScreenHeight);

            INPUT[] inputs = new INPUT[1];
            inputs[0].type = INPUT_MOUSE;
            inputs[0].mi.dx = absX;
            inputs[0].mi.dy = absY;
            inputs[0].mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE;
            inputs[0].mi.mouseData = 0;
            inputs[0].mi.time = 0;
            inputs[0].mi.dwExtraInfo = IntPtr.Zero;

            SendInput(1, inputs, Marshal.SizeOf(typeof(INPUT)));
        }

        #endregion
    }
}

