import std.algorithm;
import std.meta;
import std.random;
import std.range;
import std.stdio;
import std.typecons;

import golem;

void main()
{
	auto dataset = makeDataset();
	writeln("dataset.length: ", dataset.length);

	auto trainDataset = dataset[0 .. $ - 16];
	auto testDataset = dataset[$ - 16 .. $];

	auto model = new Model();
	auto optimizer = createOptimizer!AdaBelief(model);

	auto archiver = new ModelArchiver;
	archiver.load(model);

	foreach (epoch; 0 .. 100)
	{
		foreach (chunk; trainDataset.randomCover().chunks(16))
		{
			auto batch = batchTensor(chunk);
			auto input = batch[0];
			auto label = batch[1];

			auto output = model.forward(input, true);
			auto loss = model.loss(output.expand, label);

			optimizer.resetGrads();
			loss.backward();
			optimizer.trainStep();
		}

		{
			auto batch = batchTensor(testDataset);
			auto input = batch[0];
			auto label = batch[1];

			auto output = model.forward(input, false);
			auto loss = model.loss(output.expand, label);

			writeln(loss.value);
		}
	}
	archiver.save(model);

	void printResult(size_t pos)
	{
		auto batch = batchTensor(dataset[pos .. pos + 1]);
		auto input = batch[0];
		auto label = batch[1];

		auto output = model.forward(input, false);
		auto loss = model.loss(output.expand, label);

		writeln("-----");
		writeln("input: ", input.value[0]);
		writeln("label: ", label.value[0]);
		writeln("mean: ", output[0].value);
		writeln("stdev: ", output[1].value);
		writeln("loss: ", loss.value);
	}

	writeln("-----------------------------");
	writeln("Bankruptcy of Lehman Brothers");
	writeln("-----------------------------");
	printResult(1415);
	printResult(1418);
	printResult(1419);
	printResult(1425);
	printResult(1426);
	printResult(1427);

	// Last
	writeln();
	writeln("--------------------");
	writeln("last data");
	writeln("--------------------");
	printResult(dataset.length - 1);
}


struct Record
{
	string Date;
	float USDJPY;
}

float[] readCSVData()
{
	import std.file : readText;
	import std.csv : csvReader;

	auto csvText = readText("data/fx.csv");
	auto records = csvReader!Record(csvText, null);

	import std.array : appender;

	auto buf = appender!(float[]);
	foreach (record; records)
	{
		buf.put(record.USDJPY);
	}
	return buf.data;
}

import std.typecons : Tuple;

Tuple!(float[], float[])[] makeDataset()
{
	auto rawData = readCSVData();

	auto tempInput = new float[5];
	auto tempLabel = new float[3];

	auto result = appender!(typeof(return));
	foreach (i; 0 .. rawData.length - 9)
	{
		tempInput[0 .. $] = rawData[i .. i + 5];
		tempLabel[0 .. $] = rawData[i + 5 .. i + 8];

		result.put(tuple(tempInput.dup, tempLabel.dup));
	}

	return result.data;
}

alias InputTensor = Tensor!(float, [0, 5], UseGradient.no);
alias LabelTensor = Tensor!(float, [0, 3], UseGradient.no);

Tuple!(InputTensor, LabelTensor) batchTensor(R)(R records)
{
	static Appender!(float[]) inputs;
	static Appender!(float[]) labels;

	inputs.clear();
	labels.clear();
	foreach (r; records)
	{
		inputs.put(r[0]);
		labels.put(r[1]);
	}

	auto input = new InputTensor(inputs.data);
	auto label = new LabelTensor(labels.data);

	return tuple(input, label);
}


/+
  PDF(x) = (1 / (sqrt(2 * PI * s^2))) * exp(- (x - m)^2 / (2 * s^2))
  log(PDF(x)) = 1/2 (-(x - μ)^2/σ^2 - log(2 π) - 2 log(σ))
+/

auto logLikehood(T, U)(T m, T s, U x)
{
	import std.math : PI, stdlog = log;

	auto t = m - x;
	auto t2 = t * t;
	auto s2 = s * s;
	enum LOG_2PI = -stdlog(2 * PI);
	return -0.5 * (t2 / s2 + 2 * log(s) + LOG_2PI);
}

class Model
{
	Linear!(float, 5, 3) fc_mubase;
	Linear!(float, 5, 64) fc_mu1;
	Linear!(float, 64, 3) fc_mu2;
	Linear!(float, 5, 32) fc_stdev1;
	Linear!(float, 32, 3) fc_stdev2;

	mixin NetModule;

	auto forward(T)(T x, bool isTrain)
	{
		import std.typecons: tuple;

		auto mu_base = fc_mubase(x);
		auto mu_h1 = dropout(relu(fc_mu1(x)), 0.2, isTrain);
		auto mu_h2 = fc_mu2(mu_h1);
		auto mu = mu_base + mu_h2;
		
		auto stdev_h1 = tanhExp(fc_stdev1(x));
		auto stdev_h2 = fc_stdev2(stdev_h1);
		auto stdev = softplus(stdev_h2);

		return tuple(mu, stdev);
	}

	auto loss(T, U)(T mu, T stdev, U label)
	{
		return -mean(sum(logLikehood(mu, stdev, label)));
	}
}

auto tanhExp(T)(T x)
{
	return x * tanh(exp(x));
}
