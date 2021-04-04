import std.stdio;
import std.file;
import std.path;
import std.typecons;
import std.meta;
import std.getopt;

import golem;

void main(string[] args)
{
	size_t maxEpoch = 1;
	size_t batchSize = 32;
	auto getoptResult = getopt(args, "epoch", &maxEpoch, "batch", &batchSize);
	if (getoptResult.helpWanted)
	{
		defaultGetoptPrinter("", getoptResult.options);
		return;
	}

	auto trainImages = cast(ubyte[]) std.file.read(buildNormalizedPath("mnist_data", "train-images-idx3-ubyte"));
	auto trainLabels = cast(ubyte[]) std.file.read(buildNormalizedPath("mnist_data", "train-labels-idx1-ubyte"));
	auto testImages = cast(ubyte[]) std.file.read(buildNormalizedPath("mnist_data", "t10k-images-idx3-ubyte"));
	auto testLabels = cast(ubyte[]) std.file.read(buildNormalizedPath("mnist_data", "t10k-labels-idx1-ubyte"));

	auto trainDataset = makeDataset(trainImages[16 .. $], trainLabels[8 .. $]);
	auto testDataset = makeDataset(testImages[16 .. $], testLabels[8 .. $]);

	import std.file : exists, read;

	auto model = new Model;
	auto archiver = new ModelArchiver;
	archiver.load(model);

	auto optimizer = createOptimizer!Adam(model);

	import std.datetime.stopwatch : StopWatch;

	StopWatch sw;
	foreach (epoch; 0 .. maxEpoch)
	{
		sw.reset();
		sw.start();

		import std.random : randomCover;
		import std.range : chunks;

		foreach (samples; trainDataset.randomCover().chunks(batchSize))
		{
			auto batch = batchTensor(samples);
			auto input = batch[0];
			auto label = batch[1];

			auto output = model.forward(input, true);
			auto loss = model.loss(output, label);

			optimizer.resetGrads();
			mean(loss).backward();
			optimizer.trainStep();
		}

		sw.stop();

		float totalTrainLoss = 0;
		foreach (temp; chunks(trainDataset, 128))
		{
			auto trainBatch = batchTensor(temp);
			auto trainOutput = model.forward(trainBatch[0], false);
			auto trainLoss = model.loss(trainOutput, trainBatch[1]);

			foreach (r; trainLoss.value)
				totalTrainLoss += r[0];
		}

		float totalTestLoss = 0;
		foreach (temp; chunks(testDataset, 128))
		{
			auto testBatch = batchTensor(temp);
			auto testOutput = model.forward(testBatch[0], false);
			auto testLoss = model.loss(testOutput, testBatch[1]);

			foreach (r; testLoss.value)
				totalTestLoss += r[0];
		}

		writeln("----------");
		writefln!"%s[secs/epoch]"(sw.peek().total!"seconds");
		writefln!"trainLoss : %f"(totalTrainLoss / trainDataset.length);
		writefln!"testLoss : %f"(totalTestLoss / testDataset.length);

		archiver.save(model);
	}
	

	void printResult(string name, R)(R dataset)
	{
		import std.range : chunks;
		import mir.math.sum : sum;
		import mir.ndslice : diagonal;

		auto conf = zeros!(size_t, [10, 10]);
		foreach (sample; chunks(dataset, 128))
		{
			auto trainBatch = batchTensor(sample);
			auto trainOutput = model.forward(trainBatch[0], false);
			auto mat = confusionMatrix(trainOutput[0], trainBatch[1]);
			conf.value[] += mat.value;
		}

		writeln();
		writeln("-- confusion matrix --");
		auto names = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"];
		size_t n;
		foreach (c; conf.value)
		{
			writefln!"%s : %(%5d %)"(names[n++], c);
		}

		size_t countTotal = sum(conf.value);
		size_t countTrue = sum(conf.value.diagonal);
		writefln!"%s accuracy : %f"(name, real(countTrue) / countTotal);
	}

	printResult!"train"(trainDataset);
	printResult!"test"(testDataset);
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

alias InputTensor = Tensor!(float, [0, 1, 28, 28], UseGradient.no);
alias LabelTensor = Tensor!(float, [0, 10], UseGradient.no);

Tuple!(InputTensor, LabelTensor) batchTensor(R)(R dataset)
{
	import std.array : appender;
	auto input = appender!(float[]);
	auto label = appender!(float[]);

	foreach (data; dataset)
	{
		input ~= data[0];
		label ~= data[1];
	}

	auto inputTensor = new InputTensor(input.data);
	auto labelTensor = new LabelTensor(label.data);

	return tuple(inputTensor, labelTensor);
}


class Model
{
	Conv2D!(float, 1, 8, [3, 3]) conv1;      // [1, 28, 28] => [8, 26, 26]
	Conv2D!(float, 8, 16, [3, 3]) conv2;     // [8, 26, 26] => [16, 24, 24]
	LiftPool2D!(float, 24, 24) pool1;        // [16, 24, 24] => [64, 12, 12]
	Conv2D!(float, 16 * 4, 8, [3, 3]) conv3; // [64, 12, 12] => [8, 10, 10]
	Linear!(float, 8 * 10 * 10, 64) fc1;     // [8, 10, 10] => [64]
	Linear!(float, 64, 10) fc2;              // [64] => [10]

	alias parameters = AliasSeq!(conv1, conv2, pool1, conv3, fc1, fc2);
	this()
	{
		foreach (ref p; parameters)
			p = new typeof(p);
	}

	auto forward(T)(T x, bool isTrain)
	{
		auto h1 = relu(conv1(x));
		auto h2 = relu(conv2(h1));
		auto h3 = pool1.liftUp(h2);
		auto h4 = relu(conv3(h3[0]));
		auto h5 = relu(fc1(flatten(h4)));
		auto h6 = dropout(h5, 0.2, isTrain);
		auto h7 = fc2(h6);
		return tuple(h7, h3[1], h3[2]);
	}

	auto loss(T, U)(T output, U label)
	{
		auto loss1 = softmaxCrossEntropy(output[0], label);
		auto loss2 = mse(output[1].expand);
		auto loss3 = mse(output[2].expand);
		return loss1 + (loss2 + loss3);
	}
}

auto mse(T, U)(T x, U y)
{
	auto t = y - x;
	return mean(t * t);
}
