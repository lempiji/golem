module golem.random;

import golem.tensor;
import golem.util;

import mir.ndslice;

Tensor!(T, Shape, useGradient) uniform(T, size_t[] Shape, UseGradient useGradient = UseGradient.yes)() if (Shape.length > 0)
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

    static if (useGradient)
    {
        return new Tensor!(T, Shape)(t.sliced(Shape), null);
    }
    else
    {
        return new Tensor!(T, Shape, UseGradient.no)(t.sliced(Shape));
    }
}

unittest
{
    auto x = uniform!(float, [2, 3])();
    static assert(canBackward!(typeof(x)));

    assert(x.shape == [2, 3]);
}

unittest
{
    auto x = uniform!(float, [2, 3], UseGradient.no)();
    static assert(!canBackward!(typeof(x)));

    assert(x.shape == [2, 3]);
}

Tensor!(T, Shape, useGradient) uniform(T, size_t[] Shape, UseGradient useGradient = UseGradient.yes)(size_t size) if (Shape.length > 0 && Shape[0] == 0)
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

    static if (useGradient)
    {
        return new Tensor!(T, Shape)(t.sliced([size, expandShape!(Shape[1 .. $])]), null);
    }
    else
    {
        return new Tensor!(T, Shape, UseGradient.no)(t.sliced([size, expandShape!(Shape[1 .. $])]));
    }
}

unittest
{
    auto x = uniform!(float, [0, 4])(3);
    static assert(canBackward!(typeof(x)));

    assert(x.shape == [3, 4]);
}

unittest
{
    auto x = uniform!(float, [0, 4], UseGradient.no)(3);
    static assert(!canBackward!(typeof(x)));

    assert(x.shape == [3, 4]);
}

alias randn = uniform;


Tensor!(T, Shape, UseGradient.no) normal(T, size_t[] Shape)(in T location = 0.0, in T scale = 1.0)
if (Shape[0] != 0)
{
    import mir.ndslice : diagonal, reshape;
    import mir.random.variable : normalVar;
    import mir.random.engine : rne;

    auto result = uninitSlice!T(Shape);
    auto ngen = normalVar!T(location, scale);
    foreach (ref x; result.flattened[])
    {
        x = ngen(rne);
    }

    return new Tensor!(T, Shape, UseGradient.no)(result);
}

Tensor!(T, Shape, UseGradient.no) normal(T, size_t[] Shape)(size_t batchSize, in T location = 0.0, in T scale = 1.0)
if (Shape[0] == 0)
{
    assert(batchSize > 0);

    import mir.ndslice : diagonal, reshape;
    import mir.random.variable : normalVar;
    import mir.random.engine : rne;

    auto result = uninitSlice!T([batchSize, expandShape!(Shape[1 .. $])]);
    auto ngen = normalVar!T(location, scale);
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
    auto u = normal!(float, [2, 2])(1.0, 2.0);
    auto t = normal!(float, [0, 3])(3, 1.0, 2.0);
}

Tensor!(T, Shape, UseGradient.no) normalLike(T, size_t[] Shape, UseGradient useGrad)(Tensor!(T, Shape, useGrad) x, in T location = 0.0, in T scale = 1.0)
if (Shape[0] != 0)
{
    return normal!(T, Shape)(location, scale);
}

unittest
{
    auto x = tensor!([2, 2])([0.1, 0.2, 0.3, 0.4]);
    auto z = normalLike(x, 0, 1);

    static assert(x.staticShape == z.staticShape);
    assert(x.shape == z.shape);
}

Tensor!(T, Shape, UseGradient.no) normalLike(T, size_t[] Shape, UseGradient useGrad)(Tensor!(T, Shape, useGrad) x, in T location = 0.0, in T scale = 1.0)
if (Shape[0] == 0)
{
    return normal!(T, Shape)(x.shape[0], location, scale);
}

unittest
{
    auto x = tensor!([0, 2])([0.1, 0.2, 0.3, 0.4]);
    auto z = normalLike(x, 0, 1);

    static assert(x.staticShape == z.staticShape);
    assert(x.shape == z.shape);
}
