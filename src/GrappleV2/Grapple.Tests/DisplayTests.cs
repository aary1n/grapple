using Grapple.Core;
using Xunit;

namespace Grapple.Tests
{
    public class DisplayTests
    {
        [Fact]
        public void NormalizedToAbsolute_ZeroReturnsZero()
        {
            Assert.Equal(0, Win32Input.NormalizedToAbsolute(0.0));
        }

        [Fact]
        public void NormalizedToAbsolute_OneReturns65535()
        {
            Assert.Equal(65535, Win32Input.NormalizedToAbsolute(1.0));
        }

        [Fact]
        public void NormalizedToAbsolute_HalfReturnsMidpoint()
        {
            int result = Win32Input.NormalizedToAbsolute(0.5);
            // 0.5 * 65535 = 32767.5, truncated to 32767
            Assert.Equal(32767, result);
        }

        [Fact]
        public void NormalizedToAbsolute_ClampsNegativeToZero()
        {
            Assert.Equal(0, Win32Input.NormalizedToAbsolute(-0.5));
        }

        [Fact]
        public void NormalizedToAbsolute_ClampsAboveOneToMax()
        {
            Assert.Equal(65535, Win32Input.NormalizedToAbsolute(1.5));
        }

        [Fact]
        public void DisplayInfo_ConstructorSetsAllFields()
        {
            var info = new DisplayInfo(3840, 1080, 0, 0, 1920, 1080);

            Assert.Equal(3840, info.VirtualScreenWidth);
            Assert.Equal(1080, info.VirtualScreenHeight);
            Assert.Equal(0, info.VirtualScreenLeft);
            Assert.Equal(0, info.VirtualScreenTop);
            Assert.Equal(1920, info.PrimaryScreenWidth);
            Assert.Equal(1080, info.PrimaryScreenHeight);
        }

        [Fact]
        public void DisplayInfo_NegativeOrigin_SupportedForLeftMonitor()
        {
            // Secondary monitor to the left of primary: virtual desktop starts at -1920
            var info = new DisplayInfo(3840, 1080, -1920, 0, 1920, 1080);

            Assert.Equal(-1920, info.VirtualScreenLeft);
            Assert.Equal(3840, info.VirtualScreenWidth);
        }

        [Fact]
        public void DisplayInfo_FromSystem_ReturnsValidDimensions()
        {
            var info = DisplayInfo.FromSystem();

            // Virtual desktop must be at least as large as primary
            Assert.True(info.VirtualScreenWidth >= info.PrimaryScreenWidth);
            Assert.True(info.VirtualScreenHeight >= info.PrimaryScreenHeight);
            Assert.True(info.PrimaryScreenWidth > 0);
            Assert.True(info.PrimaryScreenHeight > 0);
        }

        [Fact]
        public void CoordinateMapping_SingleMonitor_MapsCorrectly()
        {
            // Single 1920x1080 monitor
            int virtualW = 1920, virtualH = 1080, virtualL = 0, virtualT = 0;

            double normalizedX = 0.5, normalizedY = 0.5;
            int pixelX = (int)(normalizedX * virtualW) + virtualL;
            int pixelY = (int)(normalizedY * virtualH) + virtualT;

            Assert.Equal(960, pixelX);
            Assert.Equal(540, pixelY);
        }

        [Fact]
        public void CoordinateMapping_DualMonitorSideBySide_RightMonitorReachable()
        {
            // Two 1920x1080 monitors side by side: virtual desktop = 3840x1080
            int virtualW = 3840, virtualH = 1080, virtualL = 0, virtualT = 0;

            // normalizedX=0.75 should land in the right monitor
            double normalizedX = 0.75;
            int pixelX = (int)(normalizedX * virtualW) + virtualL;

            Assert.Equal(2880, pixelX);  // 2880 = 0.75 * 3840 (right half of second monitor)
        }

        [Fact]
        public void CoordinateMapping_NegativeOrigin_MapsCorrectly()
        {
            // Secondary monitor to the left: origin at -1920
            int virtualW = 3840, virtualH = 1080, virtualL = -1920, virtualT = 0;

            // normalizedX=0.0 should land at the left edge of the left monitor
            double normalizedX = 0.0;
            int pixelX = (int)(normalizedX * virtualW) + virtualL;

            Assert.Equal(-1920, pixelX);  // Left edge of secondary monitor
        }

        [Fact]
        public void CoordinateMapping_HighDPI_PhysicalPixels()
        {
            // Single 4K monitor at 150% scaling: physical 3840x2160
            // With PerMonitorV2 DPI awareness, GetSystemMetrics returns physical pixels
            int virtualW = 3840, virtualH = 2160, virtualL = 0, virtualT = 0;

            double normalizedX = 1.0, normalizedY = 1.0;
            int pixelX = (int)(normalizedX * virtualW) + virtualL;
            int pixelY = (int)(normalizedY * virtualH) + virtualT;

            Assert.Equal(3840, pixelX);
            Assert.Equal(2160, pixelY);
        }
    }
}
