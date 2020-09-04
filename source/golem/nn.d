module golem.nn;

import golem.tensor;
import golem.random;

import mir.ndslice;

import std.meta;

enum hasParameters(T) = __traits(compiles, { auto ps = T.init.parameters; });


class Linear(T, size_t InputDim, size_t OutputDim)
{
    Tensor!(T, [InputDim, OutputDim]) weights;
    Tensor!(T, [OutputDim]) bias;

    alias parameters = AliasSeq!(weights, bias);

    this()
    {
        weights = uniform!(T, [InputDim, OutputDim]);
        bias = uniform!(T, [OutputDim]);
    }

    Tensor!(T, [Shape[0], OutputDim]) opCall(size_t[2] Shape)(Tensor!(T, Shape) x)
            if (Shape[1] == InputDim)
    {
        import golem.math : linear;

        return linear(x, this.weights, this.bias);
    }

    void resetGrads()
    {
        weights.resetGrads();
        bias.resetGrads();
    }
}

unittest
{
    import golem.math;

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