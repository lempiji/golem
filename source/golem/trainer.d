module golem.trainer;

import golem.tensor;

///
class EarlyStopping(T)
{
	immutable size_t patience = 3;

	size_t patienceStep = 0;
	T minLoss;

	this() @safe pure nothrow
	{
	}

	this(size_t patience) @safe pure nothrow
	{
		this.patience = patience;
	}

    bool shouldStop(Tensor!(T, [1]) loss) @safe @nogc nothrow
    {
        return shouldStop(loss.value[0]);
    }

    bool shouldStop(Tensor!(T, [1], UseGradient.no) loss) @safe @nogc nothrow
    {
        return shouldStop(loss.value[0]);
    }

	bool shouldStop(T loss) @safe @nogc nothrow
	{
        import std.math : isNaN;

		if (isNaN(minLoss) || loss < minLoss)
		{
			minLoss = loss;
			patienceStep = 0;
			return false;
		}

		if (++patienceStep >= patience)
		{
			return true;
		}

		return false;
	}
}

/// ditto
unittest
{
	auto es = new EarlyStopping!float;

    assert(!es.shouldStop(1.0f));
    assert(!es.shouldStop(0.9f));
    assert(!es.shouldStop(0.6f));
    assert(!es.shouldStop(0.7f));
    assert(!es.shouldStop(0.7f));
    assert(es.shouldStop(0.7f));
}

/// ditto
unittest
{
    auto es = new EarlyStopping!float(2);

    assert(!es.shouldStop(tensor!([1])([1.0f])));
    assert(!es.shouldStop(tensor!([1])([0.8f])));
    assert(!es.shouldStop(tensor!([1])([1.0f])));
    assert(es.shouldStop(tensor!([1])([1.0f])));
}
