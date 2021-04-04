import std;
import golem;

void main()
{
	auto dataset = loadData();

	randomShuffle(dataset);

	auto size = roundTo!size_t(dataset.length * 0.7);
	auto train = dataset[0 .. size];
	auto test = dataset[size .. $];

	auto model = new Model;
	auto archiver = new ModelArchiver;
	archiver.load(model);

	auto optimizer = createOptimizer!Adam(model);

	foreach (epoch; 0 .. 200)
	{
		foreach (chunk; train.randomCover().chunks(8))
		{
			auto batch = batchTensor(chunk);
			auto input = batch[0];
			auto label = batch[1];

			auto output = model.forward(input, true);
			auto loss = model.loss(output, label);

			optimizer.resetGrads();
			loss.backward();
			optimizer.trainStep();
		}

		auto trainBatch = batchTensor(train);
		auto trainOutput = model.forward(trainBatch[0], false);
		auto trainLoss = model.loss(trainOutput, trainBatch[1]);
		auto trainAcc = accuracy(trainOutput, trainBatch[1]);

		auto testBatch = batchTensor(test);
		auto testOutput = model.forward(testBatch[0], false);
		auto testLoss = model.loss(testOutput, testBatch[1]);
		auto testAcc = accuracy(testOutput, testBatch[1]);

		writefln!"epoch (%d) : %.2f / %.2f%%, %.2f / %.2f%%"(epoch + 1, trainLoss.value[0], testLoss.value[0], 100 * trainAcc, 100 * testAcc);
	}
	archiver.save(model);

	// print confusion matrix
	auto trainBatch = batchTensor(dataset);
	auto trainOutput = model.forward(trainBatch[0], false);
	auto mat = confusionMatrix(trainOutput, trainBatch[1]);

	writeln("-- confusion matrix --");
	auto names = ["Setosa    ", "Versicolor", "Virginica "];
	size_t n;
	foreach (c; mat.value)
	{
		writefln!"%s : %(%2d %)"(names[n++], c);
	}
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
alias LabelTensor = Tensor!(float, [0, 3], UseGradient.no);

Tuple!(InputTensor, LabelTensor) batchTensor(R)(R records)
{
	auto inputs = appender!(float[]);
	auto labels = appender!(float[]);

	foreach (Record data; records)
	{
		inputs.put(data.sepalLength);
		inputs.put(data.sepalWidth);
		inputs.put(data.petalLength);
		inputs.put(data.petalWidth);
		switch (data.variety)
		{
			case "Setosa":
				labels.put([1.0f, 0.0f, 0.0f]);
				break;
			case "Versicolor":
				labels.put([0.0f, 1.0f, 0.0f]);
				break;
			case "Virginica":
				labels.put([0.0f, 0.0f, 1.0f]);
				break;
			default:
				assert(false);
		}
	}

	auto inputTensor = new InputTensor(inputs.data);
	auto labelTensor = new LabelTensor(labels.data);

	return tuple(inputTensor, labelTensor);
}

class Model
{
	Linear!(float, 4, 16) fc1;
	Linear!(float, 16, 16) fc2;
	Linear!(float, 16, 16) fc3;
	Linear!(float, 16, 3) fc4;

	alias parameters = AliasSeq!(fc1, fc2, fc3, fc4);

	this()
	{
		foreach (ref p; parameters)
			p = new typeof(p);
	}

	auto forward(T)(T x, bool isTrain)
	{
		auto h1 = tanhExp(fc1(x));
		auto h2 = tanhExp(fc2(h1));
		auto h3 = dropout(h2, 0.4, isTrain);
		auto h4 = tanhExp(fc3(h3));
		auto h5 = fc4(h4);
		return h5;
	}

	auto loss(T, U)(T output, U label)
	{
		import golem.math : softmaxCrossEntropy, mean;

		return mean(softmaxCrossEntropy(output, label));
	}
}

auto tanhExp(T)(T x)
{
	import golem.math : exp, tanh;

	return x * tanh(exp(x));
}
