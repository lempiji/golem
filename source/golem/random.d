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