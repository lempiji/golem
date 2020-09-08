module golem.optimizer;

import golem.tensor;
import golem.nn;

import numir;

import std.meta;

@("train XOR")
unittest
{
    import golem.tensor;
    import golem.math;
    import golem.nn;

    // dataset
    auto inputs = tensor!([0, 2])([
            0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
            ]);
    auto labels = tensor!([0, 1])([0.0f, 1.0f, 1.0f, 0.0f,]);
    inputs.requireGrad = false;
    labels.requireGrad = false;

    // model
    auto fc1 = new Linear!(float, 2, 6);
    auto fc2 = new Linear!(float, 6, 1);

    auto optimizer = createOptimizer!SGD(fc1, fc2);

    auto forward(T)(T x)
    {
        auto h = sigmoid(fc1(x));
        auto o = sigmoid(fc2(h));
        return o;
    }

    // loss
    auto mse(T)(T output, T labels)
    {
        auto t = labels - output;
        auto t2 = t * t;
        auto l = sum(t2);
        return l;
    }

    auto lossFirst = mse(forward(inputs), labels);
    // train
    foreach (_; 0 .. 10)
    {
        auto output = forward(inputs);
        auto loss = mse(output, labels);

        optimizer.resetGrads();

        loss.backward();

        optimizer.trainStep();
    }
    auto lossLast = mse(forward(inputs), labels);

    assert(lossLast.shape == [1]);
    assert(lossLast.value[0] < lossFirst.value[0]);
}

struct SGDConfig
{
    float learningRate = 0.05;
    float momentumRate = 0.9;
}

class SGD(Params...)
{
    SGDConfig config;
    Params params;
    staticMap!(mapValue, Params) diffs;

    this(Params params)
    {
        this.params = params;
        static foreach (i; 0 .. Params.length)
        {
            this.diffs[i] = zeros_like(params[i].value);
        }
    }

    void resetGrads()
    {
        foreach (p; params)
        {
            p.resetGrads();
        }
    }

    void trainStep()
    {
        const learningRate = config.learningRate;
        const momentumRate = config.momentumRate;

        foreach (i, p; params)
        {
            diffs[i][] = momentumRate * diffs[i][] + p.grads[];
            p.value[] -= learningRate * diffs[i][];
        }
    }
}

auto createOptimizer(alias Optimizer, Params...)(Params params) if (Params.length > 0)
{
    import golem.util : staticIndexOf;

    enum firstPos = staticIndexOf!(hasParameters, Params);

    static if (firstPos != -1)
    {
        // dfmt off
        return createOptimizer!Optimizer(
            params[0 .. firstPos],
            params[firstPos].parameters,
            params[firstPos + 1 .. $]
        );
        // dfmt on
    }
    else
    {
        static if (allSatisfy!(isTensor, Params))
        {
            alias OptimizerImpl = Optimizer!(Params);
            return new OptimizerImpl(params);
        }
        else
        {
            static assert(false);
        }
    }
}

unittest
{
    class Model
    {
        Tensor!(float, [2, 2]) weight;

        alias parameters = AliasSeq!(weight);

        this()
        {
            weight = tensor!([2, 2])([1.0f, 2.0f, 3.0f, 4.0f]);
        }
    }

    auto model = new Model;
    auto optimizer = createOptimizer!SGD(model);
    assert(optimizer !is null);

    model.weight.grads[] = 1.0f;
    assert(model.weight.grads == [[1.0f, 1.0f], [1.0f, 1.0f]]);
    optimizer.resetGrads();
    assert(model.weight.grads == [[0.0f, 0.0f], [0.0f, 0.0f]]);
}

unittest
{
    import golem.nn : Linear;

    auto fc1 = new Linear!(float, 4, 4);
    auto fc2 = new Linear!(float, 4, 2);

    auto optimizer = createOptimizer!SGD(fc1, fc2);
    assert(optimizer !is null);
}


private alias mapValue(T) = T.Value;
