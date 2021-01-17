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
	auto archiver = new ModelArchiver;
	archiver.load(model);

	auto optimizer = createOptimizer!AdaBelief(model);

	import std.datetime.stopwatch : StopWatch;

	StopWatch sw;
	foreach (epoch; 0 .. 5)
	{
		sw.reset();
		sw.start();

		import std.random : randomCover;
		import std.range : chunks;

		foreach (samples; trainDataset.randomCover().chunks(256))
		{
			auto batch = batchTensor(samples);
			auto input = batch[0];
			auto label = batch[1];

			auto output = model.forward(input, true);
			auto loss = model.loss(output[0], input, output[1], output[2]);

			optimizer.resetGrads();
			loss.backward();
			optimizer.trainStep();
		}

		auto trainBatch = batchTensor(trainDataset);
		auto trainOutput = model.forward(trainBatch[0], false);
		auto trainLoss = model.loss(trainOutput[0], trainBatch[0], trainOutput[1], trainOutput[2]);

		auto testBatch = batchTensor(testDataset);
		auto testOutput = model.forward(testBatch[0], false);
		auto testLoss = model.loss(testOutput[0], testBatch[0], testOutput[1], testOutput[2]);

		sw.stop();

		const trainLossValue = trainLoss.value[0][0];
		const testLossValue = testLoss.value[0][0];
		writefln!"trainLoss/testLoss : %f / %f  %s[secs/epoch]"(trainLossValue, testLossValue, sw.peek().total!"seconds");

		archiver.save(model);
	}
	
	auto trainBatch = batchTensor(trainDataset);
	auto trainOutput = model.forward(trainBatch[0], false);
	auto labels = trainBatch[1];

	auto f = File("train_z.csv", "w");
	f.writeln("label,x,y");
	foreach (i; 0 .. labels.shape[0])
	{
		import std.algorithm : maxIndex;

		const pos = maxIndex(labels.value[i]);
		const temp = trainOutput[1].value[i];
		const x = temp[0];
		const y = temp[1];
		f.writefln!"%d,%f,%f"(pos, x, y);
	}

	saveImages(model);
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

alias InputTensor = Tensor!(float, [0, 28, 28], UseGradient.no);
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

	auto inputTensor = tensor!([0, 28, 28], UseGradient.no)(input.data);
	auto labelTensor = tensor!([0, 10], UseGradient.no)(label.data);

	return tuple(inputTensor, labelTensor);
}


class Model
{
	enum ZSize = 2;

	Linear!(float, 28 * 28, 256) fc1;
	Linear!(float, 256, 64) fc2;
	
	Linear!(float, 64, ZSize) fmu;
	Linear!(float, 64, ZSize) fvar;

	Linear!(float, ZSize, 64) decode1;
	Linear!(float, 64, 256) decode2;
	Linear!(float, 256, 28 * 28) decode3;

	alias parameters = AliasSeq!(fc1, fc2, fmu, fvar, decode1, decode2, decode3);

	this()
	{
		foreach (ref p; parameters)
			p = new typeof(p);
	}

	auto forward(InputTensor x, bool isTrain)
	{
		auto h = encode(x);
		auto z = reparam(h[0], h[1], isTrain);
		auto o = decode(z);
		return tuple(o, h[0], h[1]);
	}

	auto encode(InputTensor x)
	{
		auto h1 = relu(fc1(flatten(x)));
		auto h2 = relu(fc2(h1));
		auto mu = fmu(h2);
		auto logvar = fvar(h2);

		return tuple(mu, logvar);
	}

	auto decode(Tensor!(float, [0, ZSize]) z)
	{
		auto d1 = relu(decode1(z));
		auto d2 = relu(decode2(d1));
		auto d3 = sigmoid(decode3(d2));
		return d3;
	}

	auto reparam(Tensor!(float, [0, ZSize]) mu, Tensor!(float, [0, ZSize]) logvar, bool isTrain)
	{
		if (isTrain)
		{
			import golem.random : normal;

			return mu + normal!(float, [0, ZSize])(mu.shape[0], 0, 1) * exp(0.5 * logvar);
		}
		return mu;
	}

	auto loss(Tensor!(float, [0, 28 * 28]) output, InputTensor input, Tensor!(float, [0, ZSize]) mu, Tensor!(float, [0, ZSize]) logvar)
	{
		const c = 1.0f / output.shape[0];
		Tensor!(float, [0, 1]) loss1 = binaryCrossEntropy(output, flatten(input));
		Tensor!(float, [0, 1]) loss2 = -0.5f * sum(logvar - mu * mu - exp(logvar) + 1.0f);
		return c * (loss1 + loss2);
	}
}

auto binaryCrossEntropy(T, U)(T x, U y)
{
	auto t1 = y * log(x + 1e-7);
	auto t2 = (1 - y) * log((1.0f + 1e-7) - x);

	return -sum(t1 + t2);
}


void saveImages(Model model)
{
	import std.file : exists, mkdirRecurse;

	if (!exists("generated_images"))
		mkdirRecurse("generated_images");

	float[] ps = [-5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f];
	foreach (i, p1; ps)
	{
		foreach (j, p2; ps)
		{
			import std.format : format;

			auto filepath = format!"generated_images/gen_%d_%d.png"(i, j);
			saveAsImageFromZ(filepath, model, tensor!([0, 2])([p1, p2]));
		}
	}
}

void saveAsImageFromZ(string filepath, Model model, Tensor!(float, [0, 2]) z)
{
	auto image = model.decode(z);
	saveAsImage(filepath, image);
}

void saveAsImage(string filepath, Tensor!(float, [0, 28 * 28]) x)
{
	assert(x.shape[0] == 1);

	size_t pos = 0;
	auto pixels = new ubyte[28 * 28];
	foreach (i; 0 .. 28)
	{
		foreach (j; 0 .. 28)
		{
			import std : roundTo;

			pixels[pos] = roundTo!ubyte(ubyte.max * x.value[0, pos]);
			++pos;
		}
	}

	import imageformats;

	write_image(filepath, 28, 28, pixels, ColFmt.Y);
}
