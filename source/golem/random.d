module golem.random;

import golem.tensor;
import golem.util;

import mir.ndslice;

Tensor!(T, Shape) uniform(T, size_t[] Shape)() if (Shape.length > 0)
{
    import std.random : stduniform = uniform;
    import std.math : sqrt;

    enum size = elementSize(Shape);
    enum q = T(0.5) / sqrt(T(size));

    auto t = new T[size];
    foreach (ref x; t)
    {
        x = stduniform!"[]"(-q, q);
    }

    return new Tensor!(T, Shape)(t.sliced(Shape), null);
}

unittest
{
    auto x = uniform!(float, [2, 3])();
    assert(x.shape == [2, 3]);
}


Tensor!(T, Shape) uniform(T, size_t[] Shape)(size_t size) if (Shape.length > 0 && Shape[0] == 0)
{
    import std.random : stduniform = uniform;
    import std.math : sqrt;

    enum esize = elementSize(Shape);
    const totalSize = size * esize;
    const q = T(0.5) / sqrt(T(totalSize));

    auto t = new T[totalSize];
    foreach (ref x; t)
    {
        x = stduniform!"[]"(-q, q);
    }

    return new Tensor!(T, Shape)(t.sliced([size, expandShape!(Shape[1 .. $])]), null);
}

unittest
{
    auto x = uniform!(float, [0, 4])(3);
    assert(x.shape == [3, 4]);
}

alias randn = uniform;


Tensor!(T, Shape, UseGradient.no) normal(T, size_t[] Shape)()
if (Shape[0] != 0)
{
    import mir.ndslice : diagonal, reshape;
    import mir.random.variable : normalVar;
    import mir.random.engine : rne;

    auto result = uninitSlice!T(Shape);
    auto ngen = normalVar!T();
    foreach (ref x; result.flattened[])
    {
        x = ngen(rne);
    }

    return new Tensor!(T, Shape, UseGradient.no)(result);
}

Tensor!(T, Shape, UseGradient.no) normal(T, size_t[] Shape)(size_t batchSize)
if (Shape[0] == 0)
{
    assert(batchSize > 0);

    import mir.ndslice : diagonal, reshape;
    import mir.random.variable : normalVar;
    import mir.random.engine : rne;

    auto result = uninitSlice!T([batchSize, expandShape!(Shape[1 .. $])]);
    auto ngen = normalVar!T();
    foreach (ref x; result.flattened[])
    {
        x = ngen(rne);
    }

    return new Tensor!(T, Shape, UseGradient.no)(result);
}

unittest
{
    auto m = normal!(float, [2, 3]);
    auto n = normal!(float, [0, 4])(4);
}
