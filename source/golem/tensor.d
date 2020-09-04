module golem.tensor;

import golem.util;

import mir.ndslice;
import numir;

import std.typecons : Flag, Yes, No;

alias UseGradient = Flag!"gradient";

enum isTensor(T, U, size_t[] Shape, UseGradient hasGradient) = is(T == Tensor!(U, Shape, hasGradient));

enum isTensor(T) = is(T == Tensor!(U, Shape, flag), U, size_t[] Shape, UseGradient flag);

enum isTensor(T, UseGradient hasGradient) = is(T == Tensor!(U, Shape, hasGradient), U, size_t[] Shape);

enum isTensor(T, U, size_t[] Shape) = isTensor!(T, U, Shape, Yes.gradient) || isTensor!(T, U, Shape, No.gradient);

enum canBackward(T) = isTensor!(T, Yes.gradient);

bool testCompatibleStaticShape()(size_t[] lhsShape, size_t[] rhsShape)
{
    assert(lhsShape.length > 0);
    assert(rhsShape.length > 0);

    if (lhsShape.length != rhsShape.length)
        return false;
    
    foreach (i; 0 .. lhsShape.length)
    {
        if (i == 0 && (lhsShape[0] == 0 || rhsShape[0] == 0))
            continue;
        
        if (lhsShape[i] != rhsShape[i])
            return false;
    }
    return true;
}

unittest
{
    enum testShape = testCompatibleStaticShape([2, 2], [2, 2]);
    static assert(testShape);

    static assert(testCompatibleStaticShape([2, 3], [2, 3]));
    static assert(testCompatibleStaticShape([2, 3], [0, 3]));
    static assert(testCompatibleStaticShape([0, 3], [0, 3]));
    static assert(testCompatibleStaticShape([0, 3], [2, 3]));
    static assert(testCompatibleStaticShape([2, 3, 28, 28], [2, 3, 28, 28]));

    static assert(!testCompatibleStaticShape([2, 3], [2, 3, 3]));
    static assert(!testCompatibleStaticShape([2, 3, 3], [2, 3]));
    static assert(!testCompatibleStaticShape([1, 3], [2, 3]));
    static assert(!testCompatibleStaticShape([2, 4], [2, 3]));
}


template commonGradientType(T1, T2)
    if (isTensor!T1 && isTensor!T2)
{
    static if (canBackward!T1 || canBackward!T2)
    {
        enum commonGradientType = Yes.gradient;
    }
    else
    {
        enum commonGradientType = No.gradient;
    }
}

class Tensor(T, size_t[] Shape, UseGradient hasGradient = UseGradient.yes)
{
    alias ElementType = T;

    enum staticShape = Shape;

    static if (Shape[0] != 0)
    {
        alias shape = staticShape;
    }
    else
    {
        alias shape = runtimeShape;
    }

    alias Value = Slice!(T*, Shape.length);

    Value value;
    static if (hasGradient)
    {
        Value grads;

        bool requireGrad = true;
        size_t usedCount;
        size_t backwardCount;
        void delegate(Value grads) backwardFn;
    }

    static if (Shape[0] != 0)
    {
        this(T init)
        {
            this(slice!T(Shape, init));
        }
    }

    this(RoR)(RoR data)
    {
        this(fuse(data));
    }

    this(T[] data)
    {
        static if (Shape[0] == 0)
        {
            const batchSize = data.length / elementSize(Shape[1 .. $]);
            assert(batchSize * elementSize(Shape[1 .. $]) == data.length);
            auto value = data.dup.sliced([
                    batchSize, expandShape!(Shape[1 .. $])
                    ]);
        }
        else
        {
            auto value = data.dup.sliced(Shape);
        }

        this(value);
    }

    static if (hasGradient)
    {
        this(Value value)
        {
            this(value, null);
        }

        this(Value value, void delegate(Value grad) gradFn)
        {
            this(value, zeros_like(value), gradFn);
        }

        private this(Value value, Value grads, void delegate(Value grad) gradFn)
        {
            this.value = value;
            this.grads = grads;
            this.backwardFn = gradFn;
        }
    }
    else
    {
        this(Value value)
        {
            this.value = value;
        }
    }

    size_t[Shape.length] runtimeShape() const pure nothrow @safe @nogc
    {
        return this.value.shape;
    }

    static if (hasGradient)
    {
        void resetGrads()
        {
            grads[] = T(0);
        }

        void backward()(void delegate(ref Value grads) update)
        {
            if (requireGrad)
            {
                update(this.grads);
                ++backwardCount;

                if (backwardCount == usedCount)
                {
                    if (this.backwardFn)
                        this.backwardFn(this.grads);
                    this.usedCount = 0;
                    this.backwardCount = 0;
                }
            }
        }

        void backward(U)(U grads)
        {
            if (requireGrad)
            {
                import std.format : format;

                assert(this.grads.shape == grads.shape,
                        "%s != %s".format(this.grads.shape, grads.shape));
                this.grads[] += grads;
                ++backwardCount;

                if (backwardCount == usedCount)
                {
                    if (this.backwardFn)
                        this.backwardFn(this.grads);
                    this.usedCount = 0;
                    this.backwardCount = 0;
                }
            }
        }

        void backward()
        {
            if (requireGrad)
            {
                this.grads[] = T(1);
                if (this.backwardFn)
                    this.backwardFn(this.grads);
                this.usedCount = 0;
                this.backwardCount = 0;
            }
        }
    }

    Tensor!(T, Shape, commonGradientType!(typeof(this), RTensor)) opBinary(string op : "+", RTensor)(RTensor rhs)
        if (isTensor!RTensor)
    {
        import std.format : format;
        static assert(testCompatibleStaticShape(Shape, RTensor.staticShape), format!`Mismatch static shape %s != %s`(Shape, RTensor.staticShape));
        assert(testCompatibleStaticShape(shape, rhs.shape), format!"Mismatch runtime shape %s != %s"(shape, rhs.shape));

        auto y = slice(this.value + rhs.value);

        static if (canBackward!(typeof(this))) this.usedCount++;
        static if (canBackward!(typeof(rhs))) rhs.usedCount++;

        static if (canBackward!(typeof(this)) || canBackward!(typeof(rhs)))
        {
            return new Tensor!(T, Shape)(y, (Value grads) {
                static if (canBackward!(typeof(this))) this.backward(grads);
                static if (canBackward!(typeof(rhs))) rhs.backward(grads);
            });
        }
        else
        {
            return new Tensor!(T, Shape, No.gradient)(y);
        }
    }

    Tensor!(T, Shape, commonGradientType!(typeof(this), RTensor)) opBinary(string op : "-", RTensor)(RTensor rhs)
    {
        import std.format : format;
        static assert(testCompatibleStaticShape(Shape, RTensor.staticShape), format!`Mismatch static shape %s != %s`(Shape, RTensor.staticShape));
        assert(testCompatibleStaticShape(shape, rhs.shape), format!"Mismatch runtime shape %s != %s"(shape, rhs.shape));

        auto y = slice(this.value - rhs.value);

        static if (canBackward!(typeof(this))) this.usedCount++;
        static if (canBackward!(typeof(rhs))) rhs.usedCount++;

        static if (canBackward!(typeof(this)) || canBackward!(typeof(rhs)))
        {
            return new Tensor!(T, Shape)(y, (Value grads) {
                static if (canBackward!(typeof(this))) this.backward((ref xGrads) { xGrads[] += grads[]; });
                static if (canBackward!(typeof(rhs))) rhs.backward((ref yGrads) { yGrads[] -= grads[]; });
            });
        }
        else
        {
            return new Tensor!(T, Shape, No.gradient)(y);
        }
    }

    Tensor!(T, Shape, commonGradientType!(typeof(this), RTensor)) opBinary(string op : "*", RTensor)(RTensor rhs)
    {
        import std.format : format;
        static assert(testCompatibleStaticShape(Shape, RTensor.staticShape), format!`Mismatch static shape %s != %s`(Shape, RTensor.staticShape));
        assert(testCompatibleStaticShape(shape, rhs.shape), format!"Mismatch runtime shape %s != %s"(shape, rhs.shape));

        if (this is rhs)
        {
            auto y = slice(this.value * this.value);
            static if (canBackward!(typeof(this))) this.usedCount++;
            static if (canBackward!(typeof(this)))
            {
                return new Tensor!(T, Shape)(y, (Value grads) {
                    this.backward(2 * this.value * grads);
                });
            }
            else
            {
                return new Tensor!(T, Shape, No.gradient)(y);
            }
        }
        else
        {
            auto y = slice(this.value * rhs.value);
            static if (canBackward!(typeof(this))) this.usedCount++;
            static if (canBackward!(typeof(rhs))) rhs.usedCount++;

            static if (canBackward!(typeof(this)) || canBackward!(typeof(rhs)))
            {
                return new Tensor!(T, Shape)(y, (Value grads) {
                    static if (canBackward!(typeof(this))) this.backward(rhs.value * grads);
                    static if (canBackward!(typeof(rhs))) rhs.backward(this.value * grads);
                });
            }
            else
            {
                return new Tensor!(T, Shape, No.gradient)(y);
            }
        }
    }

    invariant()
    {
        foreach (i; 0 .. Shape.length)
        {
            if (Shape[i] != 0)
            {
                assert(Shape[i] == value.shape[i]);
                static if (hasGradient)
                    assert(Shape[i] == grads.shape[i]);
            }
            else
            {
                assert(value.shape[i] > 0);
                static if (hasGradient)
                    assert(grads.shape[i] > 0);
            }
        }
    }
}

template tensor(size_t[] Shape, UseGradient useGradient = Yes.gradient)
{
    import std.traits : isNumeric;

    Tensor!(T, Shape, useGradient) tensor(T)(T[] data)
    if (isNumeric!T)
    in(data.length > 0)
    {
        return new Tensor!(T, Shape, useGradient)(data);
    }

    Tensor!(DeepElementType!T, Shape, useGradient) tensor(T)(T[] data)
    if (!isNumeric!T)
    in(data.length > 0)
    {
        return new Tensor!(DeepElementType!T, Shape, useGradient)(data);
    }
}

unittest
{
    Tensor!(float, [2, 2]) t = tensor!([2, 2])([0.0f, 0.1f, 0.2f, 0.3f]);

    assert(t !is null);
    assert(t.staticShape == [2, 2]);
    assert(t.runtimeShape == [2, 2]);
    assert(t.shape == [2, 2]);

    static assert(isTensor!(typeof(t)));
}

unittest
{
    Tensor!(float, [0, 2]) t = tensor!([0, 2])([1.0f, 2.0f, 3.0f, 4.0f]);

    assert(t !is null);
    assert(t.staticShape == [0, 2]);
    assert(t.runtimeShape == [2, 2]);
    assert(t.shape == [2, 2]);
}

unittest
{
    Tensor!(double, [2, 2]) t = tensor!([2, 2])([[1.0, 2.0], [3.0, 4.0]]);

    assert(t !is null);
    assert(t.staticShape == [2, 2]);
    assert(t.runtimeShape == [2, 2]);
    assert(t.shape == [2, 2]);
}

unittest
{
    Tensor!(double, [0, 2]) t = tensor!([0, 2])([[1.0, 2.0], [3.0, 4.0]]);

    assert(t !is null);
    assert(t.staticShape == [0, 2]);
    assert(t.runtimeShape == [2, 2]);
    assert(t.shape == [2, 2]);
}


unittest
{
    auto x = tensor!([2, 2])([0.0f, 1.0f, 2.0f, 3.0f]);
    auto y = tensor!([2, 2])([1.0f, 2.0f, 3.0f, 4.0f]);

    auto z = x + y;
    assert(z.value[0, 0] == 1.0f);
    assert(z.value[0, 1] == 3.0f);
    assert(z.value[1, 0] == 5.0f);
    assert(z.value[1, 1] == 7.0f);
}

unittest
{
    auto a = tensor!([2, 2])([0, 1, 2, 3]);
    auto b = tensor!([3, 2])([0, 1, 2, 3, 4, 5]);
    auto c = tensor!([0, 2])([0, 1]);
    auto d = tensor!([0, 3])([0, 1, 2]);

    // dfmt off
    static assert(!__traits(compiles, { auto z = a + b; }));
    static assert( __traits(compiles, { auto z = a + c; }));
    static assert(!__traits(compiles, { auto z = a + d; }));
    static assert(!__traits(compiles, { auto z = c + d; }));
    
    static assert(!__traits(compiles, { auto z = a - b; }));
    static assert( __traits(compiles, { auto z = a - c; }));
    static assert(!__traits(compiles, { auto z = a - d; }));
    static assert(!__traits(compiles, { auto z = c - d; }));
    
    static assert(!__traits(compiles, { auto z = a * b; }));
    static assert( __traits(compiles, { auto z = a * c; }));
    static assert(!__traits(compiles, { auto z = a * d; }));
    static assert(!__traits(compiles, { auto z = c * d; }));
    // dfmt on

    import core.exception : AssertError;
    import std.exception : assertThrown;

    assertThrown!AssertError(a + c, "Mismatch runtime shape [2, 2] != [1, 2]");
    assertThrown!AssertError(a - c, "Mismatch runtime shape [2, 2] != [1, 2]");
    assertThrown!AssertError(a * c, "Mismatch runtime shape [2, 2] != [1, 2]");
}

unittest
{
    auto x = tensor!([2, 2])([0.0f, 1.0f, 2.0f, 3.0f]);
    auto y = tensor!([2, 2])([1.0f, 2.0f, 3.0f, 4.0f]);

    auto z = x - y;
    assert(z.value[0, 0] == -1.0f);
    assert(z.value[0, 1] == -1.0f);
    assert(z.value[1, 0] == -1.0f);
    assert(z.value[1, 1] == -1.0f);
}

unittest
{
    auto x = tensor!([2, 2])([-0.5f, 0.5f, 0.0f, 1.0f]);
    auto y = tensor!([2, 2])([0.5f, 0.5f, 0.5f, 0.5f]);

    auto t = tensor!([2, 2])([0.2f, 0.2f, 0.2f, 0.2f]);

    // forward
    auto z = x * y;

    // loss
    auto h = t - z;
    auto loss = h * h;

    // backward
    loss.resetGrads();
    loss.backward();

    // train
    x.value[] -= 0.1 * x.grads[];
    y.value[] -= 0.1 * y.grads[];

    auto z2 = x * y;
    auto h2 = t - z2;
    auto loss2 = h2 * h2;
    auto s = slice(loss2.value.flattened[] - loss.value.flattened[]);
    foreach (i; 0 .. 4)
    {
        assert(s[i] < 0);
    }
}


unittest
{
    Tensor!(int, [2, 2], Yes.gradient) a = tensor!([2, 2])([0, 1, 2, 3]);
    Tensor!(int, [2, 2], No.gradient) b = tensor!([2, 2], No.gradient)([0, 1, 2, 3]);

    auto x = a + b;
    auto y = a - b;
    auto z = a * b;
}

unittest
{
    Tensor!(int, [2, 2], No.gradient) a = tensor!([2, 2], No.gradient)([0, 1, 2, 3]);
    Tensor!(int, [2, 2], No.gradient) b = tensor!([2, 2], No.gradient)([0, 1, 2, 3]);

    auto x = a + b;
    auto y = a - b;
    auto z = a * b;
}
