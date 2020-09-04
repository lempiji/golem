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
