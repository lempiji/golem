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

    this(T initial)
    {
        weights = new Tensor!(T, [InputDim, OutputDim], useGradient)(initial);
        bias = new Tensor!(T, [OutputDim], useGradient)(initial);
    }

    this(T initialWeight, T initialBias)
    {
        weights = new Tensor!(T, [InputDim, OutputDim], useGradient)(initialWeight);
        bias = new Tensor!(T, [OutputDim], useGradient)(initialBias);
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
    auto fc0 = new Linear!(float, 2, 1)(-1);
    assert(fc0.weights.value[0, 0] == -1);
    assert(fc0.weights.value[1, 0] == -1);
    assert(fc0.bias.value[0] == -1);

    auto fc1 = new Linear!(float, 2, 1)(0, 1);
    assert(fc1.weights.value[0, 0] == 0);
    assert(fc1.weights.value[1, 0] == 0);
    assert(fc1.bias.value[0] == 1);

    auto fc2 = new Linear!(float, 2, 1)(1, 0);
    assert(fc2.weights.value[0, 0] == 1);
    assert(fc2.weights.value[1, 0] == 1);
    assert(fc2.bias.value[0] == 0);
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

        assert(x.shape[0] != 0);

		import std.math : sqrt;
        import golem.math : broadcastOp;

        enum eps = 1e-7;

		if (isTrain && x.shape[0] > 1)
		{
			import mir.math.sum : mirsum = sum;
			import mir.ndslice : transposed;
			import golem.util : expandIndex;

            auto tm = tempMean.value;
            tm.flattened[] = 0;
			foreach (t; x.value.ipack!1)
			{
                tm[] += t[];
			}
            tm[] /= x.shape[0];

            auto tv = tempVar.value;
			tv.flattened[] = 0;
			foreach (t; x.value.ipack!1)
			{
				tv[] += (t[] - tm[]).map!(a => a * a);
			}
            tv[] /= x.shape[0];

            this.mean.value[] = momentum * this.mean.value[] + (1 - momentum) * tm[];
			this.var.value[] = momentum * this.var.value[] + (1 - momentum) * tv[];

            this.temps.value[] = this.var.value.map!(a => sqrt(a + eps));
            tempVar.value[] = tempVar.value.map!(a => sqrt(a + eps));
            return broadcastOp!"+"(broadcastOp!"*"(broadcastOp!"-"(x, this.tempMean), factor / this.tempVar), offset);
		}

        this.temps.value[] = this.var.value.map!(a => sqrt(a + eps));
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

    import std.math : isClose;

    assert(bn.mean.value[0, 0].isClose(0.25f));
    assert(bn.mean.value[0, 1].isClose(0.35f));
    assert(bn.mean.value[1, 0].isClose(0.45f));
    assert(bn.mean.value[1, 1].isClose(0.55f));
    
    assert(bn.var.value[0, 0].isClose(0.125f));
    assert(bn.var.value[0, 1].isClose(0.125f));
    assert(bn.var.value[1, 0].isClose(0.125f));
    assert(bn.var.value[1, 1].isClose(0.125f));

    import std.math : sqrt;
    import std.conv : text;

    assert(y.value[0, 0, 0].isClose((1.0f - 2.5f) / sqrt(1.25f)), text(y.value[0, 0, 0]));
    assert(y.value[0, 0, 1].isClose((2.0f - 3.5f) / sqrt(1.25f)), text(y.value[0, 0, 1]));
    assert(y.value[0, 1, 0].isClose((3.0f - 4.5f) / sqrt(1.25f)), text(y.value[0, 1, 0]));
    assert(y.value[0, 1, 1].isClose((4.0f - 5.5f) / sqrt(1.25f)), text(y.value[0, 1, 1]));
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

    import std.math : isClose;

    assert(x.grads[0, 0, 0].isClose(2.0f));
    assert(x.grads[0, 0, 1].isClose(2.0f));
    assert(x.grads[0, 1, 0].isClose(2.0f));
    assert(x.grads[0, 1, 1].isClose(2.0f));
    assert(x.grads[1, 0, 0].isClose(2.0f));
    assert(x.grads[1, 0, 1].isClose(2.0f));
    assert(x.grads[1, 1, 0].isClose(2.0f));
    assert(x.grads[1, 1, 1].isClose(2.0f));
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

class LiftPool2D(T, size_t Height, size_t Width, UseGradient useGradient = UseGradient.yes)
{
    enum HalfH = Height / 2;
    enum HalfW = Width / 2;

    Tensor!(T, [HalfW, HalfW], useGradient) predictW;
    Tensor!(T, [HalfW, HalfW], useGradient) updateW;
    Tensor!(T, [HalfH, HalfH], useGradient) predictH;
    Tensor!(T, [HalfH, HalfH], useGradient) updateH;

    alias parameters = AliasSeq!(predictW, updateW, predictH, updateH);
    this()
    {
        import mir.ndslice : diagonal;

        // Haar wavelet
        predictW = zeros!(T, [HalfW, HalfW], useGradient)();
        predictW.value.diagonal[] = T(1);
        updateW = zeros!(T, [HalfW, HalfW], useGradient)();
        updateW.value.diagonal[] = T(0.5);

        predictH = zeros!(T, [HalfH, HalfH], useGradient)();
        predictH.value.diagonal[] = T(1);
        updateH = zeros!(T, [HalfH, HalfH], useGradient)();
        updateH.value.diagonal[] = T(0.5);
    }

    auto liftUp(U)(U x)
    if (isTensor!U && U.staticShape.length == 4 && U.staticShape[2] == Height && U.staticShape[3] == Width)
    {
        import std.typecons : tuple;
        import golem.math : splitEvenOdd2D, concat2D, projection1D;

        auto xw = splitEvenOdd2D!3(x);
        auto xw_predict = projection1D!3(xw[0], predictW);
        auto xw_d = xw[1] - xw_predict;
        auto xw_c = xw[0] + projection1D!3(xw_d, updateW);
        auto hidden = concat2D(xw_c, xw_d);

        auto xh = splitEvenOdd2D!2(hidden);
        auto xh_predict = projection1D!2(xh[0], predictH);
        auto xh_d = xh[1] - xh_predict;
        auto xh_c = xh[0] + projection1D!2(xh_d, updateH);
        auto output = concat2D(xh_c, xh_d);

        return tuple(output, tuple(xw[1], xw_predict), tuple(xh[1], xh_predict));
    }

}

unittest
{
    auto lift = new LiftPool2D!(double, 4, 4);
    auto images = tensor!([1, 1, 4, 4])([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);

    auto y = lift.liftUp(images);

    assert(y[0].shape == [1, 4, 2, 2]);

    assert(y[0].value[0, 0, 0, 0] == 3.5);
    assert(y[0].value[0, 0, 0, 1] == 5.5);
    assert(y[0].value[0, 0, 1, 0] == 11.5);
    assert(y[0].value[0, 0, 1, 1] == 13.5);
    assert(y[0].value[0, 1, 0, 0] == 1);
    assert(y[0].value[0, 1, 0, 1] == 1);
    assert(y[0].value[0, 1, 1, 0] == 1);
    assert(y[0].value[0, 1, 1, 1] == 1);
    assert(y[0].value[0, 2, 0, 0] == 4);
    assert(y[0].value[0, 2, 0, 1] == 4);
    assert(y[0].value[0, 2, 1, 0] == 4);
    assert(y[0].value[0, 2, 1, 1] == 4);
    assert(y[0].value[0, 3, 0, 0] == 0);
    assert(y[0].value[0, 3, 0, 1] == 0);
    assert(y[0].value[0, 3, 1, 0] == 0);
    assert(y[0].value[0, 3, 1, 1] == 0);

    y[0].backward();
}


class Conv2D(T, size_t C_in, size_t C_out, size_t[] kernelSize, UseGradient useGrad = UseGradient.yes)
{
    mixin Conv2DImpl!(T, C_in, C_out, kernelSize, [0, 0], useGrad);
}

class Conv2D(T, size_t C_in, size_t C_out, size_t[] kernelSize, size_t[] padding, UseGradient useGrad = UseGradient.yes)
{
    mixin Conv2DImpl!(T, C_in, C_out, kernelSize, padding, useGrad);
}

unittest
{
    import golem.random : uniform;

    auto images = uniform!(float, [1, 1, 28, 28]);
    auto conv1 = new Conv2D!(float, 1, 2, [3, 3]);
    auto y = conv1(images);
    assert(y.shape == [1, 2, 26, 26]);
    y.backward();
}

unittest
{
    import golem.random : uniform;

    auto images = uniform!(float, [1, 1, 28, 28]);
    auto conv1 = new Conv2D!(float, 1, 2, [3, 3], [1, 1]);
    auto y = conv1(images);
    assert(y.shape == [1, 2, 28, 28]);
    y.backward();
}

private mixin template Conv2DImpl(T, size_t C_in, size_t C_out, size_t[] kernelSize, size_t[] padding, UseGradient useGrad)
{
    enum size_t[] WeightShape = [C_out, C_in, kernelSize[0], kernelSize[1]];
    enum size_t[] BiasShape = [C_out];

    Tensor!(T, WeightShape, useGrad) weights;
    Tensor!(T, BiasShape, useGrad) bias;

    alias parameters = AliasSeq!(weights, bias);
    this()
    {
        import std.math : sqrt;
        import golem.random : uniform;

        weights = uniform!(T, WeightShape, useGrad)();
        bias = uniform!(T, BiasShape, useGrad)();
    }

    auto opCall(U)(U x)
    if (isTensor!U && U.staticShape.length == 4 && U.staticShape[1] == C_in)
    {
        import golem.math : conv2D;

        return conv2D!(padding)(x, weights, bias);
    }
}


class Perceptron(T, alias activateFn, size_t InputDim, size_t HiddenDim, size_t OutputDim, UseGradient useGrad = UseGradient.yes)
{
    Linear!(T, InputDim, HiddenDim, useGrad) fc1;
    Linear!(T, HiddenDim, OutputDim, useGrad) fc2;

    alias parameters = AliasSeq!(fc1, fc2);

    this()
    {
        foreach (ref p; parameters)
            p = new typeof(p);
    }

    auto opCall(U)(U x)
    {
        return fc2(activateFn(fc1(x)));
    }
}

unittest
{
    import golem.math : sigmoid;

    auto model = new Perceptron!(float, sigmoid, 2, 2, 1);
    
    auto x = tensor!([0, 2])([1.0f, 2.0f, 3.0f, 4.0f]);
    auto y = model(x);
    static assert(isTensor!(typeof(y), float, [0, 1]));

    auto z = tensor!([0, 2], UseGradient.no)([1.0f, 2.0f, 3.0f, 4.0f]);
    auto w = model(z);
    static assert(isTensor!(typeof(w), float, [0, 1]));
}

unittest
{
    import golem.math : sigmoid;

    auto model = new Perceptron!(float, sigmoid, 2, 2, 1, UseGradient.no);

    auto x = tensor!([0, 2])([1.0f, 2.0f, 3.0f, 4.0f]);
    auto y = model(x);
    static assert(isTensor!(typeof(y), float, [0, 1]));

    auto z = tensor!([0, 2], UseGradient.no)([1.0f, 2.0f, 3.0f, 4.0f]);
    auto w = model(z);
    static assert(isTensor!(typeof(w), float, [0, 1], UseGradient.no));
}
