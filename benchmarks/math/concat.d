/+ dub.sdl:
	dependency "golem" path="../.."
    dependency "lantern" version="*"
+/

module benchmarks.math.batchnorm;

import core.time;
import std.algorithm : map;
import std.datetime.stopwatch;
import golem;
import golem.random;
import lantern;

enum N = 10000;

void main()
{
    Duration[N] ds;

    auto x = randn!(float, [0, 1024])(64);
    auto y = randn!(float, [0, 1024])(64);

    StopWatch sw;
    foreach (i; 0 .. N)
    {
        sw.reset();
        sw.start();
        auto z = concat(x, y);
        sw.stop();
        ds[i] = sw.peek();
    }

    static struct Dur
    {
        Duration elapsed;
    }

    ds[].map!(t => Dur(t)).describe().printTable();
}

