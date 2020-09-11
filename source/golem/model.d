module golem.model;

import golem.tensor;
import golem.nn;
import golem.util;

import std.meta;

ubyte[] packParameters(Params...)(Params params)
{
    import golem.util : staticIndexOf;

    enum firstPos = staticIndexOf!(hasParameters, Params);

    static if (firstPos != -1)
    {
        // dfmt off
        return packParameters(
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
            import msgpack : Packer;
            import mir.ndslice : flattened, ndarray;

            Packer packer;
            packer.beginArray(params.length);
            foreach (p; params)
            {
                packer.pack(p.value.flattened[].ndarray());
            }
            return packer.stream.data;
        }
        else
        {
            static assert(false);
        }
    }
}

void unpackParameters(Params...)(ubyte[] data, ref Params params)
{
    import golem.util : staticIndexOf;

    enum firstPos = staticIndexOf!(hasParameters, Params);

    static if (firstPos != -1)
    {
        // dfmt off
        unpackParameters(
            data,
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
            import msgpack : unpack;
            import mir.ndslice : flattened, ndarray, sliced;

            auto unpacked = unpack(data);
            foreach (p; params)
            {
                assert(!unpacked.empty);
                auto temp = unpacked.front.as!(typeof(p).ElementType[]);
                assert(elementSize(p.shape) == temp.length);
                p.value = temp.sliced(p.shape);
                unpacked.popFront();
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
    auto x = tensor!([2, 3])([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    auto serializedData = packParameters(x);

    auto y = tensor!([2, 3])([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    unpackParameters(serializedData, y);

    assert(x.value == y.value);
}

unittest
{
    auto x = tensor!([2, 3])([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    auto y = tensor!([2, 2])([0.0, 0.0, 0.0, 0.0]);

    auto serializedData = packParameters(x);

    try
        unpackParameters(serializedData, y);
    catch (Throwable t)
    {
        return;
    }
    assert(false);
}

unittest
{
    auto x = tensor!([2, 2, 2])([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    auto serializedData = packParameters(x);

    auto y = tensor!([4, 2])([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    unpackParameters(serializedData, y);

    import mir.ndslice : flattened;

    assert(x.value.flattened[] == y.value.flattened[]);
}

unittest
{
    import golem.nn : Linear;
    import std.meta : AliasSeq;

    class Model
    {
        Linear!(float, 2, 2) fc1;
        Linear!(float, 2, 1) fc2;

        alias parameters = AliasSeq!(fc1, fc2);

        this()
        {
            foreach (ref p; parameters)
                p = new typeof(p);
        }
    }

    auto m1 = new Model;
    auto serializedData = packParameters(m1);

    auto m2 = new Model;
    unpackParameters(serializedData, m2);

    assert(m1.fc1.weights.value == m2.fc1.weights.value);
    assert(m1.fc1.bias.value == m2.fc1.bias.value);
    assert(m1.fc2.weights.value == m2.fc2.weights.value);
    assert(m1.fc2.bias.value == m2.fc2.bias.value);
}