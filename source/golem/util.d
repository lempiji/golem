module golem.util;

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
