module golem.util;

import mir.ndslice;

template expandShape(size_t[] Shape)
{
    import std.meta : AliasSeq;

    static if (Shape.length == 1)
    {
        alias expandShape = AliasSeq!(Shape[0]);
    }
    else
    {
        alias expandShape = AliasSeq!(Shape[0], expandShape!(Shape[1 .. $]));
    }
}

template expandIndex(size_t From, size_t To)
if (From <= To)
{
    import std.meta : AliasSeq;

    static if (From == To - 1)
        alias expandIndex = AliasSeq!(From);
    else
        alias expandIndex = AliasSeq!(From, expandIndex!(From + 1, To));
}

unittest
{
    alias s = expandIndex!(2, 4);
    static assert(s.length == 2);
    static assert(s[0] == 2);
    static assert(s[1] == 3);
}

unittest
{
    alias s = expandIndex!(3, 6);
    static assert(s.length == 3);
    static assert(s[0] == 3);
    static assert(s[1] == 4);
    static assert(s[2] == 5);
}

size_t elementSize(size_t[] shape)
{
    if (shape[0] == 0)
    {
        return elementSize(shape[1 .. $]);
    }
    size_t s = 1;
    foreach (x; shape)
    {
        s *= x;
    }
    return s;
}


package template staticIndexOf(alias F, Ts...)
{
    static if (Ts.length == 0)
    {
        enum staticIndexOf = -1;
    }
    else
    {
        enum staticIndexOf = staticIndexOfImpl!(F, 0, Ts);
    }
}

package template staticIndexOfImpl(alias F, size_t pos, Ts...)
{
    static if (Ts.length == 0)
    {
        enum staticIndexOfImpl = -1;
    }
    else
    {
        static if (F!(Ts[0]))
        {
            enum staticIndexOfImpl = pos;
        }
        else
        {
            enum staticIndexOfImpl = staticIndexOfImpl!(F, pos + 1, Ts[1 .. $]);
        }
    }
}


package auto bringToFront(size_t M, T)(T x)
if (isSlice!T)
{
    return x.transposed!(expandIndex!(T.N - M, T.N));
}

unittest
{
    import mir.ndslice;

    auto x = iota(2, 3, 4, 5);
    auto y = x.bringToFront!2;
    assert(y.shape == [4, 5, 2, 3]);
    auto z = x.bringToFront!3;
    assert(z.shape == [3, 4, 5, 2]);
}
