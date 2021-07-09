import std;
import golem;

void main()
{
	auto dataset = loadData();

	auto modelG = new Generator;
	auto modelD = new Discriminator;
	auto archiverG = new ModelArchiver("model_generator");
	auto archiverD = new ModelArchiver("model_discriminator");
	archiverG.load(modelG);
	archiverD.load(modelD);

	auto optimizerG = createOptimizer!AdaBelief(modelG);
	auto optimizerD = createOptimizer!AdaBelief(modelD);

	foreach (epoch; 0 .. 100)
	{
		// for Generator
		{
			auto generated = modelG.generate(dataset.length);
			auto judge = modelD.predict(generated[0]);
			auto loss = modelD.loss(judge, makeLabel(judge.shape[0], true));

			optimizerG.resetGrads();
			optimizerD.resetGrads();
			loss.backward();
			optimizerG.trainStep(); // train Generator only
		}

		// for Discriminator
		{
			auto generated = modelG.generate(dataset.length);
			generated[0].requireGrad = false;
			auto judgeG = modelD.predict(generated[0]);
			auto lossG = modelD.loss(judgeG, makeLabel(judgeG.shape[0], false));

			auto input = batchTensor(dataset);
			auto judgeT = modelD.predict(input);
			auto lossT = modelD.loss(judgeT, makeLabel(judgeT.shape[0], true));

			optimizerG.resetGrads();
			optimizerD.resetGrads();
			lossG.backward();
			lossT.backward();
			optimizerD.trainStep(); // train Discriminator only
		}

		{
			auto generated = modelG.generate(dataset.length);
			auto judgeG = modelD.predict(generated[0]);
			auto lossG = modelD.loss(judgeG, makeLabel(judgeG.shape[0], false));

			auto testBatch = batchTensor(dataset);
			auto judgeD = modelD.predict(testBatch);
			auto lossD = modelD.loss(judgeD, makeLabel(judgeD.shape[0], true));

			writefln!"loss: %s, %s"(lossG.value[0], lossD.value[0]);
		}
	}

	archiverG.save(modelG);
	archiverD.save(modelD);

	// save generated data
	auto f = File("iris-generated.csv", "w");
	auto t = modelG.generate(100);
	Record[] r;
	foreach (data; t[0].value)
	{
		f.writefln!"%(%s,%)"(data);
		r ~= Record(data[0], data[1], data[2], data[3], "");
	}

	import lantern;

	writeln();
	writeln("-- raw data");
	dataset.describe().printTable();
	writeln("-- generated data");
	r.describe().printTable();
}

struct Record
{
	float sepalLength;
	float sepalWidth;
	float petalLength;
	float petalWidth;
	string variety;
}

Record[] loadData()
{
	auto csvText = readText("iris.csv");

	return csvReader!Record(csvText, null).array();
}

alias InputTensor = Tensor!(float, [0, 4], UseGradient.no);

InputTensor batchTensor(R)(R records)
{
	static Appender!(float[]) data;

	data.clear();
	foreach (Record data; records)
	{
		inputs.put(data.sepalLength);
		inputs.put(data.sepalWidth);
		inputs.put(data.petalLength);
		inputs.put(data.petalWidth);
	}

	return new InputTensor(inputs.data);
}

alias LabelTensor = Tensor!(float, [0, 2], UseGradient.no);

LabelTensor makeLabel(size_t batchSize, bool isReal)
{
	auto label = zeros!(float, [0, 2])(batchSize);
	label.value[0 .. $, isReal ? 0 : 1] = 1;
	return label;
}

class Generator
{
	Linear!(float, 10, 32) fc1;
	Linear!(float, 32, 32) fc2;
	Linear!(float, 32, 4) fc3;

	mixin NetModule;

	auto generate(size_t batchSize)
	{
		import golem.random;

		auto z = normal!(float, [0, 10])(batchSize, 0, 1);
		auto h1 = dropout(sigmoid(fc1(z)), 0.5, true);
		auto h2 = dropout(sigmoid(fc2(h1)), 0.5, true);
		auto output = fc3(h2);

		return tuple(output, z);
	}
}

class Discriminator
{
	Linear!(float, 4, 32) fc1;
	Linear!(float, 32, 16) fc2;
	Linear!(float, 16, 2) fc3;

	mixin NetModule;

	auto predict(UseGradient useGrad)(Tensor!(float, [0, 4], useGrad) x)
	{
		auto h1 = sigmoid(fc1(x));
		auto h2 = sigmoid(fc2(h1));
		auto output = fc3(h2);
		return output;
	}

	auto loss(UseGradient useGrad)(Tensor!(float, [0, 2], useGrad) data, Tensor!(float, [0, 2], UseGradient.no) label)
	{
		return mean(softmaxCrossEntropy(data, label));
	}
}
