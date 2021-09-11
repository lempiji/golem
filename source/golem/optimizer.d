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

interface Optimizer
{
    void resetGrads();

    void trainStep();
}

struct SGDConfig
{
    float learningRate = 0.01;
    float momentumRate = 0.9;
    float weightDecay = 0;
}

class SGD(Params...) : Optimizer
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
        const weightDecay = config.weightDecay;

        if (momentumRate != 0 && weightDecay != 0)
        {
            foreach (i, p; params)
            {
                diffs[i][] = momentumRate * diffs[i][] + p.grads[];
                p.value[] -= learningRate * diffs[i][] + weightDecay * p.value[];
            }
        }
        else if (momentumRate != 0)
        {
            foreach (i, p; params)
            {
                diffs[i][] = momentumRate * diffs[i][] + p.grads[];
                p.value[] -= learningRate * diffs[i][];
            }
        }
        else if (weightDecay != 0)
        {
            foreach (i, p; params)
            {
                p.value[] -= learningRate * p.grads[] + weightDecay * p.value[];
            }
        }
        else
        {
            foreach (i, p; params)
            {
                p.value[] -= learningRate * p.grads[];
            }
        }
    }
}

struct AdamConfig
{
    float learningRate = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float eps = 1e-8;
    float weightDecay = 0;
}

class Adam(Params...) : Optimizer
{
    alias Values = staticMap!(mapValue, Params);

    AdamConfig config;
    Params params;
    Values ms;
    Values vs;
    size_t trainCount;

    this(Params params)
    {
        this.params = params;
        static foreach (i; 0 .. Params.length)
        {
            this.ms[i] = zeros_like(params[i].value);
            this.vs[i] = zeros_like(params[i].value);
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
        import core.math : sqrt;
        import mir.ndslice : map;

        ++trainCount;

        const learningRate = config.learningRate;
        const beta1 = config.beta1;
        const beta1_m = 1.0f - beta1;
        const c1 = 1.0f / (1.0f - beta1 ^^ trainCount);
        const beta2 = config.beta2;
        const beta2_m = 1.0f - beta2;
        const c2 = 1.0f / (1.0f - beta2 ^^ trainCount);
        const eps = config.eps;
        const weightDecay = config.weightDecay;

        foreach (i, p; params)
        {
            this.ms[i][] = beta1 * ms[i][] + beta1_m * p.grads[];
            this.vs[i][] = beta2 * vs[i][] + beta2_m * (p.grads[] * p.grads[]);
        }

        if (weightDecay != 0)
        {
            foreach (i, p; params)
            {
                auto mbar = ms[i] * c1;
                auto vbar = vs[i] * c2;

                p.value[] -= learningRate * mbar[] / vbar[].map!(a => sqrt(a + eps)) + weightDecay * p.value[];
            }
        }
        else
        {
            foreach (i, p; params)
            {
                auto mbar = ms[i] * c1;
                auto vbar = vs[i] * c2;

                p.value[] -= learningRate * mbar[] / vbar[].map!(a => sqrt(a + eps));
            }
        }
    }
}


class AdaBelief(Params...) : Optimizer
{
    alias Values = staticMap!(mapValue, Params);

    AdamConfig config;
    Params params;
    Values ms;
    Values vs;
    size_t trainCount;

    this(Params params)
    {
        this.params = params;
        static foreach (i; 0 .. Params.length)
        {
            this.ms[i] = zeros_like(params[i].value);
            this.vs[i] = zeros_like(params[i].value);
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
        import core.math : sqrt;
        import mir.ndslice : map;

        ++trainCount;

        const learningRate = config.learningRate;
        const beta1 = config.beta1;
        const beta1_m = 1.0f - beta1;
        const c1 = 1.0f / (1.0f - beta1 ^^ trainCount);
        const beta2 = config.beta2;
        const beta2_m = 1.0f - beta2;
        const c2 = 1.0f / (1.0f - beta2 ^^ trainCount);
        const eps = config.eps;
        const weightDecay = config.weightDecay;

        foreach (i, p; params)
        {
            this.ms[i][] = beta1 * ms[i][] + beta1_m * p.grads[];
            this.vs[i][] = beta2 * vs[i][] + beta2_m * (p.grads[] - ms[i][]) ^^ 2;
        }

        if (weightDecay != 0)
        {
            foreach (i, p; params)
            {
                auto mbar = ms[i] * c1;
                auto vbar = vs[i] * c2;

                p.value[] -= learningRate * mbar[] / vbar[].map!(a => sqrt(a + eps)) + weightDecay * p.value[];
            }
        }
        else
        {
            foreach (i, p; params)
            {
                auto mbar = ms[i] * c1;
                auto vbar = vs[i] * c2;

                p.value[] -= learningRate * mbar[] / vbar[].map!(a => sqrt(a + eps));
            }
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
            static if (allSatisfy!(canBackward, Params))
            {
                alias OptimizerImpl = Optimizer!(Params);
                return new OptimizerImpl(params);
            }
            else
            {
                enum trainablePos = staticIndexOf!(canNotBackward, Params);

                // dfmt off
                return createOptimizer!Optimizer(
                    params[0 .. trainablePos],
                    params[trainablePos + 1 .. $]
                );
                // dfmt on
            }
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
    auto optimizer = createOptimizer!Adam(model);
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

unittest
{
    import golem.nn : Linear;

    auto fc1 = new Linear!(float, 2, 2);
    auto fc2 = new Linear!(float, 2, 1);

    auto optimizer = createOptimizer!Adam(fc1, fc2);
    assert(optimizer !is null);
}

unittest
{
    import golem.nn : Linear;

    auto fc1 = new Linear!(float, 2, 2, UseGradient.no);
    auto fc2 = new Linear!(float, 2, 1);

    auto optimizer = createOptimizer!Adam(fc1, fc2);
    assert(optimizer !is null);
}

unittest
{
    import golem.nn : Linear;

    auto fc1 = new Linear!(float, 2, 2);
    auto fc2 = new Linear!(float, 2, 1, UseGradient.no);

    auto optimizer = createOptimizer!Adam(fc1, fc2);
    assert(optimizer !is null);
}

unittest
{
    import golem.nn : Linear, BatchNorm;

    class Model
    {
        Linear!(float, 2, 2) fc1;
        BatchNorm!(float, [2]) bn1;

        alias parameters = AliasSeq!(fc1, bn1);

        this()
        {
            foreach (ref p; parameters)
                p = new typeof(p);
        }
    }

    auto model = new Model;
    auto optimizer = createOptimizer!SGD(model);
    assert(optimizer !is null);
}

unittest
{
    enum OptimizerKind
    {
        SGD,
        Adam,
        AdaBelief,
    }

    auto fc = new Linear!(float, 2, 1)(0);

    Optimizer optimizer;
    OptimizerKind kind = OptimizerKind.Adam;

    final switch (kind)
    {
    case OptimizerKind.SGD:
        optimizer = createOptimizer!SGD(fc);
        break;
    case OptimizerKind.Adam:
        optimizer = createOptimizer!Adam(fc);
        break;
    case OptimizerKind.AdaBelief:
        optimizer = createOptimizer!AdaBelief(fc);
        break;
    }
}

private alias mapValue(T) = T.Value;

private template canNotBackward(T)
{
    enum canNotBackward = !canBackward!(T);
}
