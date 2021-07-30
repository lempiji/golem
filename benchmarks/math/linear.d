/+ dub.sdl:
	dependency "golem" path="../.."
    dependency "mir-algorithm" version="*"
    dependency "lantern" version="*"
+/
module benchmarks.math.linear;

import golem;
import golem.random;

import core.time;

void main()
{
    enum N = 20000;

    Duration[N] ds;

    auto fc = new Linear!(float, 512, 256);
    auto x = randn!(float, [0, 512])(64);

    import std.datetime.stopwatch;

    StopWatch sw;

    foreach (i; 0 .. N)
    {
        sw.reset();
        sw.start();
        auto y = fc(x);
        sw.stop();
        ds[i] = sw.peek();
    }

    import lantern;
    import std.algorithm : map;
    static struct Dur
    {
        Duration elapsed;
    }

    ds[].map!(t => Dur(t)).describe().printTable();
}

