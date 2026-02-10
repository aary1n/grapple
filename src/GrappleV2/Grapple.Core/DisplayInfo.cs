namespace Grapple.Core
{
    /// <summary>
    /// Encapsulates display geometry for coordinate mapping.
    /// Separated from Win32Input to enable unit testing without actual monitors.
    /// </summary>
    public readonly struct DisplayInfo
    {
        public readonly int VirtualScreenWidth;
        public readonly int VirtualScreenHeight;
        public readonly int VirtualScreenLeft;
        public readonly int VirtualScreenTop;
        public readonly int PrimaryScreenWidth;
        public readonly int PrimaryScreenHeight;

        public DisplayInfo(int virtualW, int virtualH, int virtualL, int virtualT, int primaryW, int primaryH)
        {
            VirtualScreenWidth = virtualW;
            VirtualScreenHeight = virtualH;
            VirtualScreenLeft = virtualL;
            VirtualScreenTop = virtualT;
            PrimaryScreenWidth = primaryW;
            PrimaryScreenHeight = primaryH;
        }

        /// <summary>
        /// Creates DisplayInfo from live system metrics.
        /// </summary>
        public static DisplayInfo FromSystem()
        {
            return new DisplayInfo(
                Win32Input.VirtualScreenWidth,
                Win32Input.VirtualScreenHeight,
                Win32Input.VirtualScreenLeft,
                Win32Input.VirtualScreenTop,
                Win32Input.ScreenWidth,
                Win32Input.ScreenHeight);
        }
    }
}
