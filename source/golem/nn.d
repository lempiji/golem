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
