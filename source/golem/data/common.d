module golem.data.common;

import std.typecons;

Tuple!(T[], T[])[N] kfold(size_t N, T)(T[] source)
out(r)
{
    size_t count;
    foreach (t; r)
    {
        assert(t[0].length + t[1].length == source.length);
        count += t[1].length;
    }
    assert(count == source.length);
}
do
{
    typeof(return) result;

    immutable len = source.length / N;
    for (size_t i = 0, pos = 0; i < N - 1; i++, pos += len)
    {
        result[i][0] = source[0 .. pos] ~ source[pos + len .. $];
        result[i][1] = source[pos .. pos + len];
    }
    result[N - 1][0] = source[0 .. len * (N - 1)];
    result[N - 1][1] = source[len * (N - 1) .. $];

    return result;
}

unittest
{
    auto dataSource = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    auto dataLoader = dataSource.kfold!5();

    assert(dataLoader.length == 5);

    assert(dataLoader[0][0] == [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    assert(dataLoader[0][1] == [1.0, 2.0]);
    
    assert(dataLoader[1][0] == [1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    assert(dataLoader[1][1] == [3.0, 4.0]);
    
    assert(dataLoader[2][0] == [1.0, 2.0, 3.0, 4.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    assert(dataLoader[2][1] == [5.0, 6.0]);
    
    assert(dataLoader[3][0] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 10.0, 11.0]);
    assert(dataLoader[3][1] == [7.0, 8.0]);
    
    assert(dataLoader[4][0] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert(dataLoader[4][1] == [9.0, 10.0, 11.0]);

    import std.parallelism: parallel;

    foreach (dataset; parallel(dataLoader[]))
    {
        auto train = dataset[0];
        auto test = dataset[1];
    }
}
