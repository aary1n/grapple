using System;

namespace Grapple.Core
{
    /// <summary>
    /// Simple exponential low-pass filter.
    /// Used internally by OneEuroFilter.
    /// </summary>
    internal class LowPassFilter
    {
        private double _y;          // Previous filtered output
        private bool _initialized;

        /// <summary>
        /// Gets the last filtered value.
        /// </summary>
        public double Last => _y;

        /// <summary>
        /// Applies exponential smoothing to the input value.
        /// </summary>
        /// <param name="value">Raw input value</param>
        /// <param name="alpha">Smoothing factor (0-1). Higher = less smoothing.</param>
        /// <returns>Filtered value</returns>
        public double Filter(double value, double alpha)
        {
            if (!_initialized)
            {
                _y = value;
                _initialized = true;
                return value;
            }

            // Exponential smoothing: y = alpha * x + (1 - alpha) * y_prev
            _y = alpha * value + (1.0 - alpha) * _y;
            return _y;
        }

        /// <summary>
        /// Resets the filter state.
        /// </summary>
        public void Reset()
        {
            _y = 0;
            _initialized = false;
        }
    }

    /// <summary>
    /// Implementation of the 1€ Filter (Casiez et al., CHI 2012).
    /// An adaptive low-pass filter that adjusts smoothing based on signal speed.
    /// Smooth when slow, responsive when fast.
    /// </summary>
    /// <remarks>
    /// Reference: http://cristal.univ-lille.fr/~casiez/1euro/
    /// </remarks>
    public class OneEuroFilter
    {
        private readonly double _minCutoff;     // Minimum cutoff frequency (Hz)
        private readonly double _beta;          // Speed coefficient
        private readonly double _dCutoff;       // Derivative cutoff frequency

        private readonly LowPassFilter _xFilter;    // Filter for the value
        private readonly LowPassFilter _dxFilter;   // Filter for the derivative

        private double _lastTimestamp;
        private double _lastValue;
        private bool _initialized;

        /// <summary>
        /// Creates a new 1€ Filter.
        /// </summary>
        /// <param name="minCutoff">Minimum cutoff frequency in Hz. Lower = smoother but more lag. Default: 1.0</param>
        /// <param name="beta">Speed coefficient. Higher = more responsive to fast movements. Default: 0.0</param>
        /// <param name="dCutoff">Derivative cutoff frequency in Hz. Default: 1.0</param>
        public OneEuroFilter(double minCutoff = 1.0, double beta = 0.0, double dCutoff = 1.0)
        {
            _minCutoff = minCutoff;
            _beta = beta;
            _dCutoff = dCutoff;

            _xFilter = new LowPassFilter();
            _dxFilter = new LowPassFilter();
        }

        /// <summary>
        /// Computes the smoothing factor alpha for a given cutoff frequency and time delta.
        /// </summary>
        private static double ComputeAlpha(double cutoff, double dt)
        {
            // tau = 1 / (2 * PI * cutoff)
            // alpha = 1 / (1 + tau / dt)
            double tau = 1.0 / (2.0 * Math.PI * cutoff);
            return 1.0 / (1.0 + tau / dt);
        }

        /// <summary>
        /// Filters a value with adaptive smoothing.
        /// </summary>
        /// <param name="value">Raw input value</param>
        /// <param name="timestamp">Time in SECONDS (not ticks!)</param>
        /// <returns>Smoothed value</returns>
        public double Filter(double value, double timestamp)
        {
            if (!_initialized)
            {
                _lastTimestamp = timestamp;
                _lastValue = value;
                _initialized = true;
                _xFilter.Filter(value, 1.0);  // Initialize with alpha=1 (no smoothing)
                _dxFilter.Filter(0.0, 1.0);   // Initialize derivative to 0
                return value;
            }

            // Compute time delta
            double dt = timestamp - _lastTimestamp;
            
            // Guard against zero or negative dt
            if (dt <= 0.0)
            {
                dt = 1.0 / 60.0;  // Assume 60fps if timestamp is bad
            }

            _lastTimestamp = timestamp;

            // Compute derivative (rate of change)
            double dx = (value - _lastValue) / dt;
            _lastValue = value;

            // Filter the derivative
            double edx = _dxFilter.Filter(dx, ComputeAlpha(_dCutoff, dt));

            // Compute adaptive cutoff based on speed
            // Faster movement = higher cutoff = less smoothing = more responsive
            double cutoff = _minCutoff + _beta * Math.Abs(edx);

            // Filter the value with adaptive alpha
            return _xFilter.Filter(value, ComputeAlpha(cutoff, dt));
        }

        /// <summary>
        /// Resets the filter state. Call when tracking is lost.
        /// </summary>
        public void Reset()
        {
            _xFilter.Reset();
            _dxFilter.Reset();
            _initialized = false;
            _lastTimestamp = 0;
            _lastValue = 0;
        }
    }
}

