module golem.tensor;

import golem.util;

import mir.ndslice;
static import numir;

import std.typecons : Flag, Yes, No;

alias UseGradient = Flag!"gradient";

enum isTensor(T, U, size_t[] Shape, UseGradient hasGradient) = is(T == Tensor!(U, Shape, hasGradient));

enum isTensor(T) = is(T == Tensor!(U, Shape, flag), U, size_t[] Shape, UseGradient flag);

enum isTensor(T, UseGradient hasGradient) = is(T == Tensor!(U, Shape, hasGradient), U, size_t[] Shape);

enum isTensor(T, U, size_t[] Shape) = isTensor!(T, U, Shape, Yes.gradient) || isTensor!(T, U, Shape, No.gradient);

enum canBackward(T) = isTensor!(T, Yes.gradient);

bool testCompatibleStaticShape(size_t[] lhsShape, size_t[] rhsShape) @safe @nogc pure nothrow
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
        enum commonGradientType = UseGradient.yes;
    }
    else
    {
        enum commonGradientType = UseGradient.no;
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
            import std.format : format;
            assert(batchSize * elementSize(Shape[1 .. $]) == data.length, format!"The number of elements in the data must match the shape of the tensor. Shape = %s, length=%s)"(Shape, data.length));
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
            this(value, numir.zeros_like(value), gradFn);
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
    
    Tensor!(T, Shape, hasGradient) opBinary(string op : "+")(T rhs)
    {
        auto y = slice(this.value[] + rhs);

        static if (canBackward!(typeof(this)))
        {
            this.usedCount++;
            return new Tensor!(T, Shape)(y, (Value grads) {
                this.backward(grads);
            });
        }
        else
        {
            return new Tensor!(T, Shape, No.gradient)(y);
        }
    }

    Tensor!(T, Shape, hasGradient) opBinaryRight(string op : "+")(T lhs)
    {
        auto y = slice(lhs + this.value[]);

        static if (canBackward!(typeof(this)))
        {
            this.usedCount++;
            return new Tensor!(T, Shape)(y, (Value grads) {
                this.backward(grads);
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

    Tensor!(T, Shape, hasGradient) opBinary(string op : "-")(T rhs)
    {
        auto y = slice(this.value[] - rhs);

        static if (canBackward!(typeof(this)))
        {
            this.usedCount++;
            return new Tensor!(T, Shape)(y, (Value grads) {
                this.backward(grads);
            });
        }
        else
        {
            return new Tensor!(T, Shape, No.gradient)(y);
        }
    }
    
    Tensor!(T, Shape, hasGradient) opBinaryRight(string op : "-")(T rhs)
    {
        auto y = slice(rhs - this.value[]);

        static if (canBackward!(typeof(this)))
        {
            this.usedCount++;
            return new Tensor!(T, Shape)(y, (Value grads) {
                this.backward((ref yGrads) { yGrads[] -= grads[]; });
            });
        }
        else
        {
            return new Tensor!(T, Shape, No.gradient)(y);
        }
    }

    Tensor!(T, Shape, commonGradientType!(typeof(this), RTensor)) opBinary(string op : "*", RTensor)(RTensor rhs)
    if (isTensor!RTensor)
    {
        import std.format : format;
        static assert(testCompatibleStaticShape(Shape, RTensor.staticShape), format!`Mismatch static shape %s != %s`(Shape, RTensor.staticShape));
        assert(testCompatibleStaticShape(shape, rhs.shape), format!"Mismatch runtime shape %s != %s"(shape, rhs.shape));

        static if (is(typeof(this) == typeof(rhs)))
        {
            if (this is rhs)
            {
                auto y = slice(this.value * this.value);
                static if (canBackward!(typeof(this)))
                {
                    this.usedCount++;
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

    Tensor!(T, Shape, hasGradient) opBinary(string op : "*")(T rhs)
    {
        auto y = slice(this.value[] * rhs);

        static if (canBackward!(typeof(this)))
        {
            this.usedCount++;

            return new typeof(this)(y, (grads) {
                this.backward(grads[] * rhs);
            });
        }
        else
        {
            return new typeof(this)(y);
        }
    }

    
    Tensor!(T, Shape, hasGradient) opBinaryRight(string op : "*")(T lhs)
    {
        auto y = slice(lhs * this.value[]);

        static if (canBackward!(typeof(this)))
        {
            this.usedCount++;

            return new typeof(this)(y, (grads) {
                this.backward(lhs * grads[]);
            });
        }
        else
        {
            return new typeof(this)(y);
        }
    }


    Tensor!(T, Shape, commonGradientType!(typeof(this), RTensor)) opBinary(string op : "/", RTensor)(RTensor rhs)
        if (isTensor!RTensor)
    {
        import std.format : format;
        static assert(testCompatibleStaticShape(Shape, RTensor.staticShape), format!`Mismatch static shape %s != %s`(Shape, RTensor.staticShape));
        assert(testCompatibleStaticShape(shape, rhs.shape), format!"Mismatch runtime shape %s != %s"(shape, rhs.shape));

        auto y = slice(this.value / rhs.value);

        static if (canBackward!(typeof(this))) this.usedCount++;
        static if (canBackward!(typeof(rhs))) rhs.usedCount++;

        static if (canBackward!(typeof(this)) || canBackward!(typeof(rhs)))
        {
            return new Tensor!(T, Shape)(y, (Value grads) {
                static if (canBackward!(typeof(this))) this.backward(grads[] / rhs.value[]);
                static if (canBackward!(typeof(rhs))) rhs.backward(-grads[] * this.value[] / (rhs.value[] * rhs.value[]));
            });
        }
        else
        {
            return new Tensor!(T, Shape, No.gradient)(y);
        }
    }

    Tensor!(T, Shape, hasGradient) opUnary(string op : "-")()
    {
        auto y = slice(-this.value[]);

        static if (hasGradient)
        {
            this.usedCount++;

            return new Tensor!(T, Shape)(y, (Value grads) {
                this.backward(-grads[]);
            });
        }
        else
        {
            return new Tensor!(T, Shape, No.gradient)(y);
        }
    }

    invariant()
    {
        import std.format : format;
        foreach (i; 0 .. Shape.length)
        {
            if (Shape[i] != 0)
            {
                assert(Shape[i] == value.shape[i], format!"size mismatched at shape[%d] (%s and %s)"(i, Shape, value.shape));
                static if (hasGradient)
                    assert(Shape[i] == grads.shape[i], format!"size mismatched at shape[%d] (%s and %s)"(i, Shape, value.shape));
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
    auto a = tensor!([2, 2])([1, 2, 3, 4]);
    auto b = tensor!([3, 2])([1, 2, 3, 4, 5, 6]);
    auto c = tensor!([0, 2])([1, 2]);
    auto d = tensor!([0, 3])([1, 2, 3]);

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
    
    static assert(!__traits(compiles, { auto z = a / b; }));
    static assert( __traits(compiles, { auto z = a / c; }));
    static assert(!__traits(compiles, { auto z = a / d; }));
    static assert(!__traits(compiles, { auto z = c / d; }));
    // dfmt on

    import core.exception : AssertError;
    import std.exception : assertThrown;

    assertThrown!AssertError(a + c, "Mismatch runtime shape [2, 2] != [1, 2]");
    assertThrown!AssertError(a - c, "Mismatch runtime shape [2, 2] != [1, 2]");
    assertThrown!AssertError(a * c, "Mismatch runtime shape [2, 2] != [1, 2]");
    assertThrown!AssertError(a / c, "Mismatch runtime shape [2, 2] != [1, 2]");
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
    auto x = tensor!([2, 2])([
        [1.0, 2.0],
        [3.0, 4.0],
    ]);
    auto y = -x;

    assert(y.value[0, 0] == -1.0);
    assert(y.value[0, 1] == -2.0);
    assert(y.value[1, 0] == -3.0);
    assert(y.value[1, 1] == -4.0);
}

unittest
{
    auto x = tensor!([0, 2], No.gradient)([
        [1.0, 2.0],
        [3.0, 4.0],
    ]);
    auto y = -x;

    assert(y.value[0, 0] == -1.0);
    assert(y.value[0, 1] == -2.0);
    assert(y.value[1, 0] == -3.0);
    assert(y.value[1, 1] == -4.0);
}

unittest
{
    auto a = tensor!([2, 2])([1.0, 2, 3, 4]);
    auto b = tensor!([0, 2])([1.0f, 2, 3, 4]);
    auto c = tensor!([2, 2], UseGradient.no)([10, 20, 30, 40]);

    auto x = a + 0.5;
    auto y = b + 0.25f;
    auto z = c + 2;

    assert(x.value[] == [[1.5, 2.5], [3.5, 4.5]]);
    assert(y.value[] == [[1.25f, 2.25f], [3.25f, 4.25f]]);
    assert(z.value[] == [[12, 22], [32, 42]]);

    assert(x.grads[] == [[0.0, 0.0], [0.0, 0.0]]);
    assert(y.grads[] == [[0.0f, 0.0f], [0.0f, 0.0f]]);
    x.backward();
    y.backward();
    assert(x.grads[] == [[1.0, 1.0], [1.0, 1.0]]);
    assert(y.grads[] == [[1.0f, 1.0f], [1.0f, 1.0f]]);

    static assert(!__traits(compiles, z.backward()));
}

unittest
{
    auto a = tensor!([2, 2])([1.0, 2, 3, 4]);
    auto b = tensor!([0, 2])([1.0f, 2, 3, 4]);
    auto c = tensor!([2, 2], UseGradient.no)([10, 20, 30, 40]);

    auto x = 0.5 + a;
    auto y = 0.25f + b;
    auto z = 2 + c;

    assert(x.value[] == [[1.5, 2.5], [3.5, 4.5]]);
    assert(y.value[] == [[1.25f, 2.25f], [3.25f, 4.25f]]);
    assert(z.value[] == [[12, 22], [32, 42]]);

    assert(x.grads[] == [[0.0, 0.0], [0.0, 0.0]]);
    assert(y.grads[] == [[0.0f, 0.0f], [0.0f, 0.0f]]);
    x.backward();
    y.backward();
    assert(x.grads[] == [[1.0, 1.0], [1.0, 1.0]]);
    assert(y.grads[] == [[1.0f, 1.0f], [1.0f, 1.0f]]);

    static assert(!__traits(compiles, z.backward()));
}

unittest
{
    auto a = tensor!([2, 2])([1.0, 2, 3, 4]);
    auto b = tensor!([0, 2])([1.0f, 2, 3, 4]);
    auto c = tensor!([2, 2], UseGradient.no)([10, 20, 30, 40]);

    auto x = a - 0.5;
    auto y = b - 0.25f;
    auto z = c - 2;

    assert(x.value[] == [[0.5, 1.5], [2.5, 3.5]]);
    assert(y.value[] == [[0.75f, 1.75f], [2.75f, 3.75f]]);
    assert(z.value[] == [[8, 18], [28, 38]]);

    assert(x.grads[] == [[0.0, 0.0], [0.0, 0.0]]);
    assert(y.grads[] == [[0.0f, 0.0f], [0.0f, 0.0f]]);
    x.backward();
    y.backward();
    assert(x.grads[] == [[1.0, 1.0], [1.0, 1.0]]);
    assert(y.grads[] == [[1.0f, 1.0f], [1.0f, 1.0f]]);

    static assert(!__traits(compiles, z.backward()));
}

unittest
{
    auto a = tensor!([2, 2])([1.0, 2, 3, 4]);
    auto b = tensor!([0, 2])([1.0f, 2, 3, 4]);
    auto c = tensor!([2, 2], UseGradient.no)([10, 20, 30, 40]);

    auto x = 0.5 - a;
    auto y = 0.25f - b;
    auto z = 2 - c;

    assert(x.value[] == [[-0.5, -1.5], [-2.5, -3.5]]);
    assert(y.value[] == [[-0.75f, -1.75f], [-2.75f, -3.75f]]);
    assert(z.value[] == [[-8, -18], [-28, -38]]);

    assert(x.grads[] == [[0.0, 0.0], [0.0, 0.0]]);
    assert(y.grads[] == [[0.0f, 0.0f], [0.0f, 0.0f]]);
    x.backward();
    y.backward();
    assert(x.grads[] == [[1.0, 1.0], [1.0, 1.0]]);
    assert(a.grads[] == [[-1.0, -1.0], [-1.0, -1.0]]);
    assert(y.grads[] == [[1.0f, 1.0f], [1.0f, 1.0f]]);
    assert(b.grads[] == [[-1.0f, -1.0f], [-1.0f, -1.0f]]);

    static assert(!__traits(compiles, z.backward()));
}

unittest
{
    auto x = tensor!([2, 2, 2])([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    auto y = 3.0 * x;
    auto z = x * 0.5;

    import std.math : isClose;

    assert(y.value[0, 0, 0].isClose(0.1 * 3.0));
    assert(y.value[0, 0, 1].isClose(0.2 * 3.0));
    assert(y.value[0, 1, 0].isClose(0.3 * 3.0));
    assert(y.value[0, 1, 1].isClose(0.4 * 3.0));
    assert(y.value[1, 0, 0].isClose(0.5 * 3.0));
    assert(y.value[1, 0, 1].isClose(0.6 * 3.0));
    assert(y.value[1, 1, 0].isClose(0.7 * 3.0));
    assert(y.value[1, 1, 1].isClose(0.8 * 3.0));
    
    assert(z.value[0, 0, 0].isClose(0.1 * 0.5));
    assert(z.value[0, 0, 1].isClose(0.2 * 0.5));
    assert(z.value[0, 1, 0].isClose(0.3 * 0.5));
    assert(z.value[0, 1, 1].isClose(0.4 * 0.5));
    assert(z.value[1, 0, 0].isClose(0.5 * 0.5));
    assert(z.value[1, 0, 1].isClose(0.6 * 0.5));
    assert(z.value[1, 1, 0].isClose(0.7 * 0.5));
    assert(z.value[1, 1, 1].isClose(0.8 * 0.5));

    y.backward();
    z.backward();

    assert(x.grads.flattened[] == [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5]);
}

unittest
{
    auto x = tensor!([2, 2])([-1.0, 0.0, 1.0, 2.0]);
    auto y = tensor!([2, 2])([-2.0, 3.0, 4.0, 5.0]);
    auto z = x / y;

    assert(z.value[0, 0] == 0.5);
    assert(z.value[0, 1] == 0.0);
    assert(z.value[1, 0] == 0.25);
    assert(z.value[1, 1] == 0.4);

    z.backward();

    assert(x.grads[0, 0] == 1.0 / -2.0);
    assert(x.grads[0, 1] == 1.0 / 3.0);
    assert(x.grads[1, 0] == 1.0 / 4.0);
    assert(x.grads[1, 1] == 1.0 / 5.0);
    
    assert(y.grads[0, 0] == 1.0 / -2.0 / -2.0);
    assert(y.grads[0, 1] == -0.0 / 3.0 / 3.0);
    assert(y.grads[1, 0] == -1.0 / 4.0 / 4.0);
    assert(y.grads[1, 1] == -2.0 / 5.0 / 5.0);
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
    Tensor!(int, [2, 2], Yes.gradient) a = tensor!([2, 2])([1, 2, 3, 4]);
    Tensor!(int, [2, 2], No.gradient) b = tensor!([2, 2], No.gradient)([1, 2, 3, 4]);

    auto x = a + b;
    auto y = a - b;
    auto z = a * b;
    auto w = a / b;
}

unittest
{
    Tensor!(int, [2, 2], No.gradient) a = tensor!([2, 2], No.gradient)([1, 2, 3, 4]);
    Tensor!(int, [2, 2], No.gradient) b = tensor!([2, 2], No.gradient)([1, 2, 3, 4]);

    auto x = a + b;
    auto y = a - b;
    auto z = a * b;
    auto w = a / b;
}

///
Tensor!(T, Shape, useGrad) zeros(T, size_t[] Shape, UseGradient useGrad = UseGradient.no)()
if (Shape[0] != 0)
{
    return new typeof(return)(numir.zeros!T(Shape));
}

///ditto
unittest
{
    auto z = zeros!(float, [2, 2]);
    assert(z.shape == [2, 2]);
    assert(z.value[0, 0] == 0);
    assert(z.value[0, 1] == 0);
    assert(z.value[1, 0] == 0);
    assert(z.value[1, 1] == 0);
}

///ditto
unittest
{
    auto z = zeros!(float, [2, 2], UseGradient.yes);
    assert(z.shape == [2, 2]);
    assert(z.value[0, 0] == 0);
    assert(z.value[0, 1] == 0);
    assert(z.value[1, 0] == 0);
    assert(z.value[1, 1] == 0);
}

///
Tensor!(T, Shape, useGrad) zeros(T, size_t[] Shape, UseGradient useGrad = UseGradient.no)(size_t batchSize)
if (Shape[0] == 0)
{
    return new typeof(return)(numir.zeros!T([batchSize, expandShape!(Shape[1 .. $])]));
}

///ditto
unittest
{
    auto z = zeros!(float, [0, 2])(2);
    assert(z.shape == [2, 2]);
    assert(z.value[0, 0] == 0);
    assert(z.value[0, 1] == 0);
    assert(z.value[1, 0] == 0);
    assert(z.value[1, 1] == 0);
}

///ditto
unittest
{
    auto z = zeros!(float, [0, 2], UseGradient.yes)(2);
    assert(z.shape == [2, 2]);
    assert(z.value[0, 0] == 0);
    assert(z.value[0, 1] == 0);
    assert(z.value[1, 0] == 0);
    assert(z.value[1, 1] == 0);
}

///
Tensor!(T, Shape, UseGradient.no) zerosLike(T, size_t[] Shape, UseGradient useGradient)(Tensor!(T, Shape, useGradient) x)
{
    static if (x.staticShape[0] == 0)
    {
        return zeros!(T, Shape)(x.shape[0]);
    }
    else
    {
        return zeros!(T, Shape)();
    }
}

///ditto
Tensor!(T, Shape, useGrad) zerosLike(UseGradient useGrad, T, size_t[] Shape, UseGradient useGradient)(Tensor!(T, Shape, useGradient) x)
{
    static if (x.staticShape[0] == 0)
    {
        return zeros!(T, Shape, useGrad)(x.shape[0]);
    }
    else
    {
        return zeros!(T, Shape, useGrad)();
    }
}

///ditto
unittest
{
    auto x = tensor!([2, 2])([1.0f, 2.0f, 3.0f, 4.0f]);
    auto x1 = zerosLike(x);

    assert(x.shape == x1.shape);
    assert(x1.value == zeros!(float, [2, 2]).value);
    static assert(!canBackward!(typeof(x1)));
}

///ditto
unittest
{
    auto x = tensor!([2, 3])([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]);
    auto x1 = zerosLike(x);

    assert(x.shape == x1.shape);
    assert(x1.value == zeros!(float, [2, 3]).value);
    static assert(!canBackward!(typeof(x1)));
}

///ditto
unittest
{
    auto x = tensor!([2, 3])([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]);
    auto x1 = zerosLike!(UseGradient.yes)(x);

    static assert(canBackward!(typeof(x1)));
}


///
Tensor!(T, Shape, useGrad) ones(T, size_t[] Shape, UseGradient useGrad = UseGradient.no)()
if (Shape[0] != 0)
{
    return new typeof(return)(numir.ones!T(Shape));
}

///ditto
Tensor!(T, Shape, useGrad) ones(T, size_t[] Shape, UseGradient useGrad = UseGradient.no)(size_t batchSize)
if (Shape[0] == 0)
{
    return new typeof(return)(numir.ones!T([batchSize, expandShape!(Shape[1 .. $])]));
}

///ditto
unittest
{
    auto o = ones!(float, [2, 2]);
    assert(!canBackward!(typeof(o)));
    assert(o.shape == [2, 2]);
    assert(o.value[0, 0] == 1);
    assert(o.value[0, 1] == 1);
    assert(o.value[1, 0] == 1);
    assert(o.value[1, 1] == 1);
}

///ditto
unittest
{
    auto o = ones!(float, [0, 2])(2);
    assert(!canBackward!(typeof(o)));
    assert(o.shape == [2, 2]);
    assert(o.value[0, 0] == 1);
    assert(o.value[0, 1] == 1);
    assert(o.value[1, 0] == 1);
    assert(o.value[1, 1] == 1);
}

///ditto
unittest
{
    auto o = ones!(float, [2, 3], UseGradient.yes);
    static assert(canBackward!(typeof(o)));
}

///
Tensor!(T, Shape, UseGradient.no) onesLike(T, size_t[] Shape, UseGradient useGradient)(Tensor!(T, Shape, useGradient) x)
{
    static if (x.staticShape[0] == 0)
    {
        return ones!(T, Shape)(x.shape[0]);
    }
    else
    {
        return ones!(T, Shape)();
    }
}

///ditto
Tensor!(T, Shape, useGrad) onesLike(UseGradient useGrad, T, size_t[] Shape, UseGradient useGradient)(Tensor!(T, Shape, useGradient) x)
{
    static if (x.staticShape[0] == 0)
    {
        return ones!(T, Shape, useGrad)(x.shape[0]);
    }
    else
    {
        return ones!(T, Shape, useGrad)();
    }
}

///ditto
unittest
{
    auto x = tensor!([2, 2])([1.0f, 2.0f, 3.0f, 4.0f]);
    auto x1 = onesLike(x);

    assert(x.shape == x1.shape);
    assert(x1.value == ones!(float, [2, 2]).value);
}

///ditto
unittest
{
    auto x = tensor!([2, 3])([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]);
    auto x1 = onesLike(x);

    assert(x.shape == x1.shape);
    assert(x1.value == ones!(float, [2, 3]).value);
}

///ditto
unittest
{
    auto x = tensor!([2, 3])([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]);
    auto x1 = onesLike!(UseGradient.yes)(x);

    static assert(canBackward!(typeof(x1)));
}

///
Tensor!(T, TargetShape, useGrad) reshape(size_t[] TargetShape, T, size_t[] Shape, UseGradient useGrad)(Tensor!(T, Shape, useGrad) x)
{
    static if (Shape[0] == 0 && TargetShape[0] == 0)
    {
        const batchSize = x.shape[0];
        const ptrdiff_t[TargetShape.length] runtimeTargetShape = [batchSize, expandShape!(TargetShape[1 .. $])];
    }
    else
    {
        static if (Shape[0] != 0)
            enum batchSize = Shape[0];
        else
            enum batchSize = TargetShape[0];
        enum ptrdiff_t[TargetShape.length] runtimeTargetShape = [batchSize, expandShape!(TargetShape[1 .. $])];
    }

    import mir.ndslice : reshape;

	int err;
	auto yValue = reshape(x.value, runtimeTargetShape, err);

    static if (useGrad)
    {
        x.usedCount++;
        return new Tensor!(T, TargetShape)(yValue, (grads) {
            x.backward((ref xGrads) {
                xGrads.flattened[] += grads.flattened[];
            });
        });
    }
    else
    {
        return new Tensor!(T, TargetShape, useGrad)(yValue);
    }
}

/// ditto
unittest
{
    auto x = tensor!([0, 2, 2])([1, 2, 3, 4]);
    auto y = x.reshape!([0, 4]);

    assert(y.shape == [1, 4]);
    assert(y.value[0, 0] == 1);
    assert(y.value[0, 1] == 2);
    assert(y.value[0, 2] == 3);
    assert(y.value[0, 3] == 4);

    assert(x.grads == [[[0, 0], [0, 0]]]);
    y.backward();
    assert(x.grads == [[[1, 1], [1, 1]]]);
}

/// ditto
unittest
{
    auto x = tensor!([0, 4], UseGradient.no)([1, 2, 3, 4]);
    auto y = x.reshape!([1, 2, 2]);

    assert(y.shape == [1, 2, 2]);
    assert(y.value[0, 0, 0] == 1);
    assert(y.value[0, 0, 1] == 2);
    assert(y.value[0, 1, 0] == 3);
    assert(y.value[0, 1, 1] == 4);

    static assert(!canBackward!(typeof(y)));
}
