using System;
using System.Runtime.InteropServices;

namespace Grapple.Core
{
    /// <summary>
    /// P/Invoke wrapper for Windows cursor control.
    /// Uses SetCursorPos for reliable absolute cursor positioning.
    /// </summary>
    public static class Win32Input
    {
        #region Constants

        private const int SM_CXSCREEN = 0;   // Primary monitor width
        private const int SM_CYSCREEN = 1;   // Primary monitor height

        #endregion

        #region P/Invoke

        [DllImport("user32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool SetCursorPos(int x, int y);

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
        /// <returns>True if successful, false otherwise</returns>
        public static bool MoveMouse(int x, int y)
        {
            return SetCursorPos(x, y);
        }

        #endregion
    }
}

