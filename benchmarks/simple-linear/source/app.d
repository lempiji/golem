import golem;
import golem.random;

import core.time;
import std.getopt;

void main(string[] args)
{
	size_t batchSize = 64;
	getopt(args, "size", &batchSize);

	enum N = 20_000;
	enum InputDim = 512;
	enum OutputDim = 256;

	Duration[N] ds;

	auto fc = new Linear!(float, InputDim, OutputDim);
	auto x = randn!(float, [0, InputDim])(batchSize);

	import std.datetime.stopwatch;

	StopWatch sw;
	float total = 0;
	foreach (i; 0 .. N)
	{
		sw.reset();
		sw.start();
		auto y = relu(fc(x));
		y.backward();
		sw.stop();

		import mir.math.sum : sum;

		total += sum(y.value);
		ds[i] = sw.peek();
	}

	import lantern;
	import std.algorithm : map;

	static struct Dur
	{
		Duration elapsed;
	}

	ds[].map!(t => Dur(t)).describe().printTable();

	import std.stdio : writeln;

	writeln(total);
}
