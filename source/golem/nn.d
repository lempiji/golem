module golem.nn;

import golem.tensor;
import golem.random;

import mir.ndslice;

import std.meta;

enum hasParameters(T) = __traits(compiles, { auto ps = T.init.parameters; });


class Linear(T, size_t InputDim, size_t OutputDim, UseGradient useGradient = UseGradient.yes)
{
    Tensor!(T, [InputDim, OutputDim], useGradient) weights;
    Tensor!(T, [OutputDim], useGradient) bias;

    alias parameters = AliasSeq!(weights, bias);

    this()
    {
        weights = uniform!(T, [InputDim, OutputDim], useGradient);
        bias = uniform!(T, [OutputDim], useGradient);
    }

    auto opCall(size_t[2] Shape, UseGradient useGradient)(Tensor!(T, Shape, useGradient) x)
            if (Shape[1] == InputDim)
    {
        import golem.math : linear;

        return linear(x, this.weights, this.bias);
    }

    static if (useGradient)
    {
        void resetGrads()
        {
            weights.resetGrads();
            bias.resetGrads();
        }
    }
}

unittest
{
    import golem.math : flatten;

    auto fc1 = new Linear!(float, 2 * 2, 1);

    Tensor!(float, [3, 2, 2]) x = new Tensor!(float, [3, 2, 2])(0.0f);
    Tensor!(float, [3, 4]) b = flatten(x);
    Tensor!(float, [3, 1]) y = fc1(b);
    assert(y.value.shape == [3, 1]);
}

unittest
{
    auto fc = new Linear!(float, 2, 1);
    static assert(hasParameters!(typeof(fc)));
}

unittest
{
    auto fc1 = new Linear!(float, 2, 1, UseGradient.yes);
    auto fc2 = new Linear!(float, 2, 1, UseGradient.no);

    auto x = new Tensor!(float, [2, 2], UseGradient.yes)(1.0f);
    auto y = new Tensor!(float, [2, 2], UseGradient.no)(1.0f);

    auto a = fc1(x);
    auto b = fc1(y);
    auto c = fc2(x);
    auto d = fc2(y);

    static assert(canBackward!(typeof(a)));
    static assert(canBackward!(typeof(b)));
    static assert(canBackward!(typeof(c)));
    static assert(!canBackward!(typeof(d)));
}

class BatchNorm(T, size_t[] Shape, UseGradient useGrad = UseGradient.yes)
{
	Tensor!(T, Shape, UseGradient.no) mean;
	Tensor!(T, Shape, UseGradient.no) var;
	Tensor!(T, Shape, useGrad) factor;
	Tensor!(T, Shape, useGrad) offset;

    Tensor!(T, Shape, UseGradient.no) tempMean;
	Tensor!(T, Shape, UseGradient.no) tempVar;
	Tensor!(T, Shape, UseGradient.no) temps;
	T momentum = 0.9;

	alias parameters = AliasSeq!(mean, var, factor, offset);

	this()
	{
		mean = zeros!(T, Shape);
		var = zeros!(T, Shape);
		factor = ones!(T, Shape, useGrad);
		offset = zeros!(T, Shape, useGrad);
        tempMean = zeros!(T, Shape);
		tempVar = zeros!(T, Shape);
		temps = zeros!(T, Shape);
	}

	auto opCall(size_t[] ShapeX, UseGradient useGradX)(Tensor!(T, ShapeX, useGradX) x, bool isTrain)
	{
		static assert(ShapeX[1 .. $] == Shape);

		import std.math : sqrt;
        import golem.math : broadcastOp;

		if (isTrain && x.shape[0] != 0)
		{
			import mir.math.sum : mirsum = sum;
			import mir.ndslice : transposed;
			import golem.util : expandIndex;

            enum eps = 1e-7;
			immutable batchFactor = T(1.0) / x.shape[0];

            tempMean.value.flattened[] = 0;
			foreach (t; x.value.ipack!1)
			{
                tempMean.value[] += t;
			}
            tempMean.value[] *= batchFactor;

			tempVar.value.flattened[] = 0;
			foreach (t; x.value)
			{
				tempVar.value[] += (t[] - tempMean.value[]) ^^ 2;
			}
            tempVar.value[] *= batchFactor;

            this.mean.value[] = momentum * this.mean.value[] + (1 - momentum) * tempMean.value[];
			this.var.value[] = momentum * this.var.value[] + (1 - momentum) * tempVar.value[];

            this.temps.value[] = this.var.value.map!(a => sqrt(a + 1e-7));
            tempVar.value[] = tempVar.value.map!(a => sqrt(a + 1e-7));
            return broadcastOp!"+"(broadcastOp!"*"(broadcastOp!"-"(x, this.tempMean), factor / this.tempVar), offset);
		}

        this.temps.value[] = this.var.value.map!(a => sqrt(a + 1e-7));
		return broadcastOp!"+"(broadcastOp!"*"(broadcastOp!"-"(x, this.mean), factor / this.temps), offset);
	}
}

unittest
{
    auto x = tensor!([0, 2, 2])([
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 3.0f, 4.0f, 5.0f,
        3.0f, 4.0f, 5.0f, 6.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
    ]);

    auto bn = new BatchNorm!(float, [2, 2]);
    auto y = bn(x, true);

    import std.math : approxEqual;

    assert(bn.mean.value[0, 0].approxEqual(0.25f));
    assert(bn.mean.value[0, 1].approxEqual(0.35f));
    assert(bn.mean.value[1, 0].approxEqual(0.45f));
    assert(bn.mean.value[1, 1].approxEqual(0.55f));
    
    assert(bn.var.value[0, 0].approxEqual(0.125f));
    assert(bn.var.value[0, 1].approxEqual(0.125f));
    assert(bn.var.value[1, 0].approxEqual(0.125f));
    assert(bn.var.value[1, 1].approxEqual(0.125f));

    import std.math : sqrt;
    import std.conv : text;

    assert(y.value[0, 0, 0].approxEqual((1.0f - 2.5f) / sqrt(1.25f)), text(y.value[0, 0, 0]));
    assert(y.value[0, 0, 1].approxEqual((2.0f - 3.5f) / sqrt(1.25f)), text(y.value[0, 0, 1]));
    assert(y.value[0, 1, 0].approxEqual((3.0f - 4.5f) / sqrt(1.25f)), text(y.value[0, 1, 0]));
    assert(y.value[0, 1, 1].approxEqual((4.0f - 5.5f) / sqrt(1.25f)), text(y.value[0, 1, 1]));
}

unittest
{
    auto x = tensor!([0, 2, 2])([
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 3.0f, 4.0f, 5.0f,
    ]);

    auto bn = new BatchNorm!(float, [2, 2]);
    auto y = bn(x, true);

    import std.conv : text;

    assert(x.grads[] == [[[0.0f, 0.0f], [0.0f, 0.0f]], [[0.0f, 0.0f], [0.0f, 0.0f]]], text(x.grads));
    y.backward();

    import std.math : approxEqual;

    assert(x.grads[0, 0, 0].approxEqual(2.0f));
    assert(x.grads[0, 0, 1].approxEqual(2.0f));
    assert(x.grads[0, 1, 0].approxEqual(2.0f));
    assert(x.grads[0, 1, 1].approxEqual(2.0f));
    assert(x.grads[1, 0, 0].approxEqual(2.0f));
    assert(x.grads[1, 0, 1].approxEqual(2.0f));
    assert(x.grads[1, 1, 0].approxEqual(2.0f));
    assert(x.grads[1, 1, 1].approxEqual(2.0f));
}

struct Activation(alias f)
{
    import std.functional : unaryFun;

    alias fun = unaryFun!f;

    auto opCall(T)(T x)
    {
        return fun(x);
    }
}

unittest
{
    import golem.math : sigmoid, tanh;

    Activation!sigmoid f1;
    Activation!tanh f2;

    auto x = tensor!([2, 2])([1.0f, 2.0f, 3.0f, 4.0f]);
    
    auto a = f1(x);
    auto b = f2(x);
    auto c = sigmoid(x);
    auto d = tanh(x);

    assert(a.value == c.value);
    assert(b.value == d.value);
}


class Sequence(Ts...)
{
    Ts layers;

    private alias isNetModule(alias m) = hasParameters!(typeof(m));

    static if (Filter!(isNetModule, AliasSeq!(layers)).length > 0)
    {
        alias parameters = Filter!(isNetModule, AliasSeq!(layers));

        this()
        {
            foreach (ref p; parameters)
                p = new typeof(p);
        }
    }


    auto opCall(T)(T x)
    {
        return opCall!0(x);
    }

    private auto opCall(size_t n, T)(T x)
    {
        static if (n == Ts.length)
        {
            return x;
        }
        else
        {
            return opCall!(n + 1)(layers[n](x));
        }
    }
}

unittest
{
    import golem.math : sigmoid;

    auto net = new Sequence!(
        Linear!(float, 2, 2),
        Activation!sigmoid,
        Linear!(float, 2, 2),
        Activation!sigmoid,
        Linear!(float, 2, 1),
        Activation!sigmoid,
    );

    static assert(hasParameters!(typeof(net)));

    auto x = tensor!([0, 2])([1.0f, 2.0f, 3.0f, 4.0f]);
    auto y = net(x);
}

unittest
{
    import golem.math : sigmoid;

    auto net = new Sequence!(
        Activation!sigmoid,
        Activation!sigmoid,
        Activation!sigmoid,
    );

    static assert(!hasParameters!(typeof(net)));
}
