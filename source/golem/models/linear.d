module golem.models.linear;

import golem.math : sigmoid;
import golem.tensor : Tensor, UseGradient;
import golem.nn : Linear;
import golem.optimizer : Adam, SGD, AdaBelief, createOptimizer;
import golem.trainer :  EarlyStopping;
import std.typecons : Tuple, tuple;
import std.meta : AliasSeq;
import std.array : array;
import mir.ndslice : ndarray;

struct LogisticFitOptions
{
    size_t maxEpoch = 500;
    float penaltyWeightsDecay = 1e-3;
}

class LogisticRegression(T, size_t InputDim, size_t OutputDim, UseGradient useGrad = UseGradient
        .yes)
{
    private alias InputTensor = Tensor!(T, [0, InputDim]);
    private alias OutputTensor = Tensor!(T, [0, OutputDim]);

    Linear!(T, InputDim, OutputDim, useGrad) weights;

    alias parameters = AliasSeq!(weights);

    this()
    {
        weights = new typeof(weights)(T(0));
    }

    void fit(in Tuple!(T[], T[])[] train, in Tuple!(T[], T[])[] test, LogisticFitOptions options = LogisticFitOptions
            .init)
    {
        import std.stdio;

        auto optimizer = createOptimizer!AdaBelief(weights);
        optimizer.config.weightDecay = options.penaltyWeightsDecay;
        auto stopper = new EarlyStopping!T();

        auto dataset_train = makeTensors(train);
        auto dataset_test = makeTensors(test);
        foreach (epoch; 0 .. options.maxEpoch)
        {
            auto y_train = forward(dataset_train[0]);
            auto loss_train = calculateLoss(y_train, dataset_train[1]);

            optimizer.resetGrads();
            loss_train.backward();
            optimizer.trainStep();

            auto y_test = forward(dataset_test[0]);
            auto loss_test = calculateLoss(y_test, dataset_test[1]);

            if (stopper.shouldStop(loss_test))
                break;
        }
    }

    void save(string modelDirPath)
    {
        import golem : ModelArchiver;

        auto archiver = new ModelArchiver(modelDirPath);
        archiver.save(weights);
    }

    void load(string modelDirPath)
    {
        import golem : ModelArchiver;

        auto archiver = new ModelArchiver(modelDirPath);
        archiver.load(weights);
    }

    T[] predict(T[] input)
    in
    {
        assert(input.length == InputDim);
    }
    do
    {
        auto inputTensor = new Tensor!(T, [1, InputDim], UseGradient.no)(input);
        auto output = forward(inputTensor);

        return output.value[0].array();
    }

    T[][] predict(T[][] inputs)
    in
    {
        assert(inputs.length % InputDim == 0);
    }
    do
    {
        auto inputTensor = new Tensor!(T, [0, InputDim], UseGradient.no)(inputs);
        auto output = forward(inputTensor);

        return output.value.ndarray();
    }

    private auto makeTensors(in Tuple!(T[], T[])[] dataset)
    {
        import std.array : appender;

        immutable size = dataset.length;
        auto inputBuf = new T[size * InputDim];
        auto labelBuf = new T[size * OutputDim];
        for (size_t i = 0, inputPos = 0, labelPos = 0; i < size; i++, inputPos += InputDim, labelPos += OutputDim)
        {
            auto temp = dataset[i];
            inputBuf[inputPos .. inputPos + InputDim] = temp[0];
            labelBuf[labelPos .. labelPos + OutputDim] = temp[1];
        }

        auto inputTensor = new InputTensor(inputBuf);
        auto labelTensor = new OutputTensor(labelBuf);

        return tuple(inputTensor, labelTensor);
    }

    private auto forward(U)(U x)
    {
        return sigmoid(weights(x));
    }

    private auto calculateLoss(U, V)(U output, V label)
    {
        import golem : sum;

        auto temp = label - output;
        static if (OutputDim == 1)
            return sum(temp * temp);
        else
            return sum(sum(temp * temp));
    }
}

unittest
{
    Tuple!(float[], float[])[] dataset = [
        tuple([20.0f, 0.0f, 1], [0.0f]),
        tuple([24.0f, 0.5f, 1], [0.0f]),
        tuple([30.0f, 0.0f, 0], [0.0f]),
        tuple([36.0f, 1.0f, 0], [0.0f]),
        tuple([48.0f, 1.0f, 1], [0.0f]),
        tuple([58.0f, 1.0f, 1], [1.0f]),
        tuple([55.0f, 1.5f, 1], [1.0f]),
        tuple([18.0f, 0.5f, 0], [0.0f]),
        tuple([24.0f, 0.5f, 0], [0.0f]),
        tuple([30.0f, 0.0f, 1], [0.0f]),
        tuple([34.0f, 1.0f, 1], [0.0f]),
        tuple([50.0f, 1.5f, 0], [0.0f]),
        tuple([64.0f, 1.0f, 0], [1.0f]),
        tuple([57.0f, 2.0f, 1], [1.0f]),
    ];

    auto model = new LogisticRegression!(float, 3, 1);

    model.fit(dataset, dataset);

    assert(model.predict(dataset[0][0])[0] < 0.5);
    assert(model.predict(dataset[5][0])[0] > 0.5);
}

unittest
{
    Tuple!(float[], float[])[] dataset = [
        tuple([20.0f, 0.0f, 1], [0.0f, 0]),
        tuple([24.0f, 0.5f, 1], [0.0f, 0]),
        tuple([30.0f, 0.0f, 0], [0.0f, 0]),
        tuple([36.0f, 1.0f, 0], [0.0f, 1]),
        tuple([48.0f, 1.0f, 1], [0.0f, 0]),
        tuple([58.0f, 1.0f, 1], [1.0f, 0]),
        tuple([55.0f, 1.5f, 1], [1.0f, 0]),
        tuple([18.0f, 0.5f, 0], [0.0f, 0]),
        tuple([24.0f, 0.5f, 0], [0.0f, 0]),
        tuple([30.0f, 0.0f, 1], [0.0f, 0]),
        tuple([34.0f, 1.0f, 1], [0.0f, 0]),
        tuple([50.0f, 1.5f, 0], [0.0f, 1]),
        tuple([64.0f, 1.0f, 0], [1.0f, 1]),
        tuple([57.0f, 2.0f, 1], [1.0f, 0]),
    ];

    auto model = new LogisticRegression!(float, 3, 2);

    model.fit(dataset, dataset);

    assert(model.predict(dataset[0][0])[0] < 0.5);
    assert(model.predict(dataset[5][0])[0] > 0.5);
    assert(model.predict(dataset[11][0])[1] > 0.5);
}
