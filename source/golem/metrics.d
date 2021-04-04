module golem.metrics;

import golem.tensor;

/// calc accuracy for 1 class (single 0-1 output)
float accuracy(T, size_t[] Shape, UseGradient useGrad1, UseGradient useGrad2)(Tensor!(T, Shape, useGrad1) output, Tensor!(T, Shape, useGrad2) label)
if (Shape.length == 2 && Shape[1] == 1)
in(output.shape[0] == label.shape[0])
{
	import std.algorithm : maxIndex;

    const batchSize = output.shape[0];

	size_t trueAnswer;
	foreach (i; 0 .. batchSize)
	{
        const x = output.value[i, 0] > 0.5;
        const y = label.value[i, 0] > 0.5;
        if (x == y) ++trueAnswer;
	}

	return float(trueAnswer) / float(batchSize);
}

/// ditto
unittest
{
    auto xt = tensor!([0, 1])([0.1, 0.9, 0.8, 0.2]);
    auto y0 = tensor!([0, 1])([1.0, 0.0, 0.0, 1.0]); // true: 0
    auto y1 = tensor!([0, 1])([1.0, 0.0, 1.0, 1.0]); // true: 1
    auto y2 = tensor!([0, 1])([1.0, 1.0, 1.0, 1.0]); // true: 2
    auto y3 = tensor!([0, 1])([1.0, 1.0, 1.0, 0.0]); // true: 3
    auto y4 = tensor!([0, 1])([0.0, 1.0, 1.0, 0.0]); // true: 4

    assert(accuracy(xt, y0) == 0.0f);
    assert(accuracy(xt, y1) == 0.25f);
    assert(accuracy(xt, y2) == 0.5f);
    assert(accuracy(xt, y3) == 0.75f);
    assert(accuracy(xt, y4) == 1.0f);
}


/// calc accuracy for multi class (multiple 0-1 output)
float accuracy(T, size_t[] Shape, UseGradient useGrad1, UseGradient useGrad2)(Tensor!(T, Shape, useGrad1) output, Tensor!(T, Shape, useGrad2) label)
if (Shape.length == 2 && Shape[1] > 1)
in(output.shape[0] == label.shape[0])
{
	import std.algorithm : maxIndex;

    const batchSize = output.shape[0];

	size_t trueAnswer;
	foreach (i; 0 .. batchSize)
	{
		const x = maxIndex(output.value[i, 0 .. $]);
		const y = maxIndex(label.value[i, 0 .. $]);
		if (x == y) ++trueAnswer;
	}

	return float(trueAnswer) / float(batchSize);
}

/// ditto
unittest
{
    auto xt = tensor!([0, 2])([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6], [0.2, 0.8]]);
    auto y0 = tensor!([0, 2])([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]); // true: 0
    auto y1 = tensor!([0, 2])([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]); // true: 1
    auto y2 = tensor!([0, 2])([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]); // true: 2
    auto y3 = tensor!([0, 2])([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]); // true: 3
    auto y4 = tensor!([0, 2])([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]); // true: 4

    assert(accuracy(xt, y0) == 0.0f);
    assert(accuracy(xt, y1) == 0.25f);
    assert(accuracy(xt, y2) == 0.5f);
    assert(accuracy(xt, y3) == 0.75f);
    assert(accuracy(xt, y4) == 1.0f);
}

Tensor!(size_t, [Shape[1], Shape[1]], UseGradient.no) confustionMatrix(T, size_t[] Shape, UseGradient useGrad1, UseGradient useGrad2)(Tensor!(T, Shape, useGrad1) x, Tensor!(T, Shape, useGrad2) y)
if (Shape.length == 2 && Shape[1] > 1)
{
    assert(x.shape[0] == y.shape[0]);

    import std.algorithm : maxIndex;
    import mir.ndslice;

    auto result = slice!size_t([Shape[1], Shape[1]], 0);
    foreach (i; 0 .. x.shape[0])
    {
        const xindex = maxIndex(x.value[i]);
        const yindex = maxIndex(y.value[i]);
        result[yindex, xindex]++;
    }
    return new Tensor!(size_t, [Shape[1], Shape[1]], UseGradient.no)(result);
}

unittest
{
    auto x = tensor!([0, 4])([
        [0.0f, 0.0f, 0.0f, 1.0f],
        [0.0f, 0.0f, 1.0f, 0.0f],
        [0.0f, 1.0f, 0.0f, 0.0f],
        [1.0f, 0.0f, 0.0f, 0.0f],
        [0.0f, 0.0f, 0.0f, 1.0f],
        [0.0f, 0.0f, 1.0f, 0.0f],
        [0.0f, 1.0f, 0.0f, 0.0f],
        [1.0f, 0.0f, 0.0f, 0.0f],
        [0.0f, 0.0f, 0.0f, 1.0f],
        [0.0f, 0.0f, 1.0f, 0.0f],
        [0.0f, 1.0f, 0.0f, 0.0f],
        [1.0f, 0.0f, 0.0f, 0.0f],
        [0.0f, 0.0f, 0.0f, 1.0f],
        [0.0f, 0.0f, 1.0f, 0.0f],
        [0.0f, 1.0f, 0.0f, 0.0f],
        [1.0f, 0.0f, 0.0f, 0.0f],
    ]);
    
    auto y = tensor!([0, 4])([
        [0.0f, 0.0f, 0.0f, 1.0f],
        [0.0f, 0.0f, 1.0f, 0.0f],
        [0.0f, 1.0f, 0.0f, 0.0f],
        [1.0f, 0.0f, 0.0f, 0.0f],
        [0.0f, 0.0f, 1.0f, 0.0f],
        [0.0f, 1.0f, 0.0f, 0.0f],
        [1.0f, 0.0f, 0.0f, 0.0f],
        [0.0f, 0.0f, 0.0f, 1.0f],
        [0.0f, 1.0f, 0.0f, 0.0f],
        [1.0f, 0.0f, 0.0f, 0.0f],
        [0.0f, 0.0f, 0.0f, 1.0f],
        [0.0f, 0.0f, 1.0f, 0.0f],
        [1.0f, 0.0f, 0.0f, 0.0f],
        [0.0f, 0.0f, 0.0f, 1.0f],
        [0.0f, 0.0f, 1.0f, 0.0f],
        [0.0f, 1.0f, 0.0f, 0.0f],
    ]);

    auto m = confustionMatrix(x, y);
    static import numir;

    assert(m.value == numir.ones!size_t(4, 4));
}
