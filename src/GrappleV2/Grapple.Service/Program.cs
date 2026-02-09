using Grapple.Core;
using Grapple.Service;

var config = GrappleConfigLoader.Load();

var builder = Host.CreateApplicationBuilder(args);
builder.Services.AddSingleton(config);
builder.Services.AddHostedService<Worker>();

var host = builder.Build();
host.Run();
