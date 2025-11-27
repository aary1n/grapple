using System.Threading;
using System.Threading.Tasks;

namespace Grapple.Nodes
{
    /// <summary>
    /// Defines the lifecycle of a node in the graph.
    /// </summary>
    public interface IGraphNode
    {
        /// <summary>
        /// Starts the processing loop for the node.
        /// Should return immediately and run processing on a separate thread/task.
        /// Uses ValueTask to prevent Task allocation overhead on hot paths.
        /// </summary>
        /// <param name="ct">Cancellation token to signal shutdown.</param>
        ValueTask StartAsync(CancellationToken ct);
    }
}

