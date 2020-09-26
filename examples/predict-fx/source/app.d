import std;
import golem;

void main()
{
	auto rawdata = readCSVData();
	auto dataset = makeDataset(rawdata);

	auto predict = dataset[$ - 1 .. $];
	dataset = dataset[0 .. $ - 1];

	auto train = dataset[0 .. $ - 10];
	auto test = dataset[$ - 10 .. $];

	auto model = new Model();
	auto optimizer = createOptimizer!Adam(model);

	foreach (epoch; 0 .. 100)
	{
		foreach (chunk; train.randomCover().chunks(16))
		{
			auto batch = batchTensor(chunk);
			auto input = batch[0];
			auto label = batch[1];

			auto output = model.forward(input);
			auto loss = model.loss(output, label);

			optimizer.resetGrads();
			loss.backward();
			optimizer.trainStep();
		}

		auto trainBatch = batchTensor(train);
		auto trainLoss = model.loss(model.forward(trainBatch[0]), trainBatch[1]);
		auto testBatch = batchTensor(test);
		auto testLoss = model.loss(model.forward(testBatch[0]), testBatch[1]);
		writefln!"loss (%d) : %s / %s"(epoch, trainLoss.value, testLoss.value);
	}

	auto predictBatch = batchTensor(predict);
	auto predictOutput = model.forward(predictBatch[0]);
	writeln(predictOutput.value[0]);
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

enum InputDays = 5;
enum PredictDays = 3;

Tuple!(float[], float[])[] makeDataset(float[] records)
out (results)
{
	foreach (r; results)
	{
		assert(r[0].length == InputDays);
		assert(r[1].length == PredictDays);
	}
}
do
{
	import std.array : appender;

	auto buf = appender!(typeof(return));
	foreach (i; InputDays .. records.length - PredictDays)
	{
		buf.put(tuple(records[i - InputDays .. i], records[i .. i + PredictDays]));
	}
	buf.put(tuple(records[$ - InputDays .. $], new float[](PredictDays)));
	return buf.data;
}

Tuple!(Tensor!(float, [0, InputDays]), Tensor!(float, [0, PredictDays])) batchTensor(Records)(
		Records records)
{
	import std.array : appender;

	auto inputs = appender!(float[]);
	auto labels = appender!(float[]);

	foreach (record; records)
	{
		inputs.put(record[0]);
		labels.put(record[1]);
	}

	auto inputTensor = tensor!([0, InputDays])(inputs.data);
	auto labelTensor = tensor!([0, PredictDays])(labels.data);
	return tuple(inputTensor, labelTensor);
}

class Model
{
	Linear!(float, InputDays, 100) fc1;
	Linear!(float, 100, PredictDays) fc2;

	alias parameters = AliasSeq!(fc1, fc2);

	this()
	{
		foreach (ref p; parameters)
			p = new typeof(p);
	}

	auto forward(T)(T x)
	{
		auto h1 = relu(fc1(x));
		return fc2(h1);
	}

	auto loss(T, U)(T output, U label)
	{
		auto t = label - output;
		return mean(sum(t * t));
	}
}
