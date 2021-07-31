import std.stdio;
import std.typecons;
import std.random;
import std.range : chunks;
import std.algorithm : map;
import std.array;

import golem;
import golem.util;

void main()
{
	auto dataset = [
		tuple([0.0f, 0.0f], [0.0f]),
		tuple([0.0f, 1.0f], [1.0f]),
		tuple([1.0f, 0.0f], [1.0f]),
		tuple([1.0f, 1.0f], [0.0f]),
	];

	auto net = new Model();
	auto optimizer = createOptimizer!SGD(net);
	
	foreach (epoch; 0 .. 20_000)
	{
		foreach (data; dataset.randomCover().chunks(2))
		{
			auto t = batchTensor!([2], [1])(data.array());
			auto x = t[0];
			auto label = t[1];

			auto y = net.forward(x);
			auto loss = net.loss(y, label);

			optimizer.resetGrads();
			loss.backward();
			optimizer.trainStep();
		}
	}

	foreach (data; dataset)
	{
		auto x = tensor!([1, 2])(data[0]);
		auto y = net.forward(x);

		writeln(x.value, " -> ", y.value, " : ", data[1]);
	}
}

import std.meta;

class Model
{
	Linear!(float, 2, 2) fc1;
	Linear!(float, 2, 1) fc2;

	mixin NetModule;

	auto forward(size_t[2] Shape)(Tensor!(float, Shape) x)
	if (Shape[1] == 2)
	{
		auto h = sigmoid(fc1(x));
		auto o = sigmoid(fc2(h));
		return o;
	}

	auto loss(Tensor!(float, [0, 1]) y, Tensor!(float, [0, 1]) label)
	{
		auto t = label - y;
		auto s = t * t;
		return sum(s);
	}
}

Tuple!(Tensor!(T, [0, expandShape!InputShape]), Tensor!(T, [0, expandShape!LabelShape])) batchTensor(size_t[] InputShape, size_t[] LabelShape, T)(Tuple!(T[], T[])[] dataset)
out(r; r[0].shape[0] == r[1].shape[0])
{
	import std.array : Appender;

	static Appender!(T[]) dataBuf;
	static Appender!(T[]) labelBuf;
	
	dataBuf.clear();
	labelBuf.clear();
	foreach(data; dataset)
	{
		dataBuf.put(data[0]);
		labelBuf.put(data[1]);
	}

	auto data = new Tensor!(T, [0, expandShape!InputShape])(dataBuf.data);
	auto label = new Tensor!(T, [0, expandShape!LabelShape])(labelBuf.data);

	return tuple(data, label);
}

unittest
{
	auto dataset = [
		tuple([0.0f, 0.0f], [0.0f]),
		tuple([0.0f, 1.0f], [1.0f]),
		tuple([1.0f, 0.0f], [1.0f]),
		tuple([1.0f, 1.0f], [0.0f]),
	];

	auto t = batchTensor!([2], [1])(dataset);
	auto x = t[0];
	auto label = t[1];

	assert(x.shape == [4, 2]);
	assert(label.shape == [4, 1]);
}

unittest
{
	auto m = new Model();
	static assert(hasParameters!(typeof(m)));
}
