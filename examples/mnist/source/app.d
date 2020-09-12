import std.stdio;
import std.file;
import std.path;
import std.typecons;
import std.meta;

import golem;

void main()
{
	auto trainImages = cast(ubyte[]) std.file.read(buildNormalizedPath("mnist_data", "train-images-idx3-ubyte"));
	auto trainLabels = cast(ubyte[]) std.file.read(buildNormalizedPath("mnist_data", "train-labels-idx1-ubyte"));
	auto testImages = cast(ubyte[]) std.file.read(buildNormalizedPath("mnist_data", "t10k-images-idx3-ubyte"));
	auto testLabels = cast(ubyte[]) std.file.read(buildNormalizedPath("mnist_data", "t10k-labels-idx1-ubyte"));

	auto trainDataset = makeDataset(trainImages[16 .. $], trainLabels[8 .. $]);
	auto testDataset = makeDataset(testImages[16 .. $], testLabels[8 .. $]);

	import std.file : exists, read;

	auto model = new Model;
	if (exists("model.dat"))
		unpackParameters(cast(ubyte[]) read("model.dat"), model);

	auto optimizer = createOptimizer!SGD(model);

	import std.datetime.stopwatch : StopWatch;

	StopWatch sw;
	foreach (epoch; 0 .. 5)
	{
		sw.reset();
		sw.start();

		import std.random : randomCover;
		import std.range : chunks;

		foreach (samples; trainDataset.randomCover().chunks(32))
		{
			auto batch = batchTensor(samples);
			auto input = batch[0];
			auto label = batch[1];

			auto output = model.forward(input);
			auto loss = model.loss(output, label);

			optimizer.resetGrads();
			loss.backward();
			optimizer.trainStep();
		}

		auto trainBatch = batchTensor(trainDataset);
		auto trainOutput = model.forward(trainBatch[0]);
		auto trainLoss = model.loss(trainOutput, trainBatch[1]);

		auto testBatch = batchTensor(testDataset);
		auto testOutput = model.forward(testBatch[0]);
		auto testLoss = model.loss(testOutput, testBatch[1]);

		sw.stop();
		
		writeln("----------");
		writefln!"trainLoss/testLoss : %f / %f  %s[secs/epoch]"(trainLoss.value[0], testLoss.value[0], sw.peek().total!"seconds");
		writefln!"train accuracy %.2f%%"(100 * accuracy(trainOutput, trainBatch[1]));
		writefln!"test accuracy %.2f%%"(100 * accuracy(testOutput, testBatch[1]));

		import std.file : fwrite = write;

		fwrite("model.dat", packParameters(model));
	}
}


Tuple!(float[], float[])[] makeDataset(ubyte[] images, ubyte[] labels)
in(images.length == labels.length * 784)
{
	import std.array : appender;

	auto result = appender!(Tuple!(float[], float[])[]);

	float[784] input;
	float[10] label;
	label[] = 0;

	enum max = cast(float) ubyte.max;
	foreach (i; 0 .. labels.length)
	{
		input[] = images[i * 784 .. i * 784 + 784][] / max;
		label[labels[i]] = 1;
		scope (success) label[labels[i]] = 0;

		result.put(tuple(input.dup, label.dup));
	}

	return result.data;
}

Tuple!(Tensor!(float, [0, 28, 28]), Tensor!(float, [0, 10], UseGradient.no)) batchTensor(R)(R dataset)
{
	import std.array : appender;
	auto input = appender!(float[]);
	auto label = appender!(float[]);

	foreach (data; dataset)
	{
		input ~= data[0];
		label ~= data[1];
	}

	auto inputTensor = tensor!([0, 28, 28])(input.data);
	auto labelTensor = tensor!([0, 10], UseGradient.no)(label.data);

	return tuple(inputTensor, labelTensor);
}


class Model
{
	Linear!(float, 28 * 28, 100) fc1;
	Linear!(float, 100, 10) fc2;

	alias parameters = AliasSeq!(fc1, fc2);

	this()
	{
		foreach (ref p; parameters)
			p = new typeof(p);
	}

	auto forward(T)(T x)
	{
		auto h1 = relu(fc1(flatten(x)));
		auto h2 = relu(fc2(h1));
		return h2;
	}

	auto loss(T, U)(T output, U label)
	{
		return softmaxCrossEntropy(output, label);
	}
}
