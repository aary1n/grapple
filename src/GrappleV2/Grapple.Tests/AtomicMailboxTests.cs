using System;
using Grapple.Core;
using Xunit;

namespace Grapple.Tests
{
    public class AtomicMailboxTests : IDisposable
    {
        // Use unique signal names per test to avoid cross-test interference
        private readonly AtomicMailbox _mailbox;

        public AtomicMailboxTests()
        {
            _mailbox = new AtomicMailbox($"Local\\GrappleTest_{Guid.NewGuid():N}");
        }

        public void Dispose()
        {
            _mailbox.Dispose();
        }

        [Fact]
        public void RegisterConsumer_Once_Succeeds()
        {
            _mailbox.RegisterConsumer();
            // No exception = success
        }

        [Fact]
        public void RegisterConsumer_Twice_ThrowsInvalidOperation()
        {
            _mailbox.RegisterConsumer();
            Assert.Throws<InvalidOperationException>(() => _mailbox.RegisterConsumer());
        }

        [Fact]
        public void UnregisterConsumer_AllowsReregistration()
        {
            _mailbox.RegisterConsumer();
            _mailbox.UnregisterConsumer();
            _mailbox.RegisterConsumer();  // Should not throw
        }

        [Fact]
        public void PublishConsume_BasicRoundtrip()
        {
            _mailbox.RegisterConsumer();

            int dropped = _mailbox.Publish(42);
            Assert.Equal(-1, dropped);  // Nothing was in the slot

            int consumed = _mailbox.Consume();
            Assert.Equal(42, consumed);
        }

        [Fact]
        public void Publish_DropsOldValue_ReturnsDroppedId()
        {
            _mailbox.RegisterConsumer();

            _mailbox.Publish(1);
            int dropped = _mailbox.Publish(2);

            Assert.Equal(1, dropped);  // Frame 1 was dropped

            int consumed = _mailbox.Consume();
            Assert.Equal(2, consumed);  // Only latest frame available
        }

        [Fact]
        public void Consume_WhenEmpty_ReturnsNegativeOne()
        {
            _mailbox.RegisterConsumer();

            int consumed = _mailbox.Consume();
            Assert.Equal(-1, consumed);
        }

        [Fact]
        public void WaitForData_WithTimeout_ReturnsFalseOnTimeout()
        {
            _mailbox.RegisterConsumer();

            // Should timeout immediately (1ms)
            bool signaled = _mailbox.WaitForData(1);
            Assert.False(signaled);
        }

        [Fact]
        public void WaitForData_AfterPublish_ReturnsTrue()
        {
            _mailbox.RegisterConsumer();

            _mailbox.Publish(99);
            bool signaled = _mailbox.WaitForData(1000);
            Assert.True(signaled);
        }
    }
}
