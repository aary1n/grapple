using System;
using System.Runtime.InteropServices;

namespace Grapple.Core
{
    /// <summary>
    /// P/Invoke wrapper for Windows cursor and mouse button control.
    /// Uses SetCursorPos for positioning and mouse_event for button clicks.
    /// </summary>
    public static class Win32Input
    {
        #region Constants

        private const int SM_CXSCREEN = 0;   // Primary monitor width
        private const int SM_CYSCREEN = 1;   // Primary monitor height

        // mouse_event flags
        private const uint MOUSEEVENTF_LEFTDOWN = 0x0002;
        private const uint MOUSEEVENTF_LEFTUP = 0x0004;

        // Virtual key codes
        public const int VK_F9 = 0x78;

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
        /// <returns>True if successful, false otherwise</returns>
        public static bool MoveMouse(int x, int y)
        {
            return SetCursorPos(x, y);
        }

        /// <summary>
        /// Sends a left mouse button down event at the current cursor position.
        /// </summary>
        public static void LeftDown()
        {
            mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, UIntPtr.Zero);
        }

        /// <summary>
        /// Sends a left mouse button up event at the current cursor position.
        /// </summary>
        public static void LeftUp()
        {
            mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, UIntPtr.Zero);
        }

        /// <summary>
        /// Checks if a key is currently pressed.
        /// Uses GetAsyncKeyState which works even when the app doesn't have focus.
        /// </summary>
        /// <param name="vKey">Virtual key code (e.g., VK_F9)</param>
        /// <returns>True if the key is currently held down</returns>
        public static bool IsKeyDown(int vKey)
        {
            // High bit (0x8000) indicates key is currently down
            return (GetAsyncKeyState(vKey) & 0x8000) != 0;
        }

        #endregion
    }
}

