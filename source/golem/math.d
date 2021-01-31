module golem.math;

import golem.tensor;
import golem.util;

import mir.ndslice;

import std.typecons : No;

version (all) // exp
{
    Tensor!(T, Shape, useGradient) exp(T, size_t[] Shape, UseGradient useGradient)(Tensor!(T, Shape, useGradient) x)
    {
        import std.math : stdexp = exp;

        auto y = slice(x.value.map!(a => stdexp(a)));

        static if (canBackward!(typeof(x)))
        {
            x.usedCount++;

            return new Tensor!(T, Shape)(y, (Slice!(T*, Shape.length) grads) {
                x.backward(y * grads);
            });
        }
        else
        {
            return new Tensor!(T, Shape, No.gradient)(y);
        }
    }

    unittest
    {
        auto x = tensor!([2])([-1.0f, 1.0f]);
        auto y = exp(x);

        import std.math : stdexp = exp, approxEqual;

        assert(y.value[0].approxEqual(stdexp(-1.0f)));
        assert(y.value[1].approxEqual(stdexp(1.0f)));

        y.resetGrads();
        y.backward();

        import std : format;

        assert(x.grads[0].approxEqual(y.value[0]), "%s : %s".format(x.grads[0], y.value[0]));
        assert(x.grads[1].approxEqual(y.value[1]), "%s : %s".format(x.grads[1], y.value[1]));
    }

    unittest
    {
        auto x = tensor!([2, 2, 2])([
                0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f
                ]);

        auto y = flatten(x);

        assert(y.shape == [2, 4]);

        auto z = exp(y);
        z.backward();

        int err;
        assert(x.grads == z.value.reshape([2, 2, 2], err));
    }

    unittest
    {
        auto x = tensor!([2, 2], No.gradient)([1.0, 2.0, 3.0, 4.0]);
        auto y = exp(x);
        
        import std.math : stdexp = exp, approxEqual;

        assert(y.value[0, 0].approxEqual(stdexp(1.0f)));
        assert(y.value[0, 1].approxEqual(stdexp(2.0f)));
        assert(y.value[1, 0].approxEqual(stdexp(3.0f)));
        assert(y.value[1, 1].approxEqual(stdexp(4.0f)));
    }
}

version (all) // log
{
    Tensor!(T, Shape, useGradient) log(T, size_t[] Shape, UseGradient useGradient)(Tensor!(T, Shape, useGradient) x)
    {
        import std.math : stdlog = log;
        import mir.ndslice : slice, map;

        auto y = slice(x.value.map!(a => T(stdlog(a))));

        static if (useGradient)
        {
            x.usedCount++;

            alias Return = typeof(return);
            alias Value = Return.Value;
            return new Return(y, (Value grads) {
                x.backward((ref xGrads) { xGrads[] += grads[] / x.value[]; });
            });
        }
        else
        {
            return new typeof(return)(y);
        }
    }

    unittest
    {
        auto x = tensor!([0, 2])([
            [1.0, 2.0],
            [3.0, 4.0],
        ]);
        auto y = log(x);

        import std.math : stdlog = log, approxEqual;

        assert(y.value[0, 0].approxEqual(stdlog(1.0)));
        assert(y.value[0, 1].approxEqual(stdlog(2.0)));
        assert(y.value[1, 0].approxEqual(stdlog(3.0)));
        assert(y.value[1, 1].approxEqual(stdlog(4.0)));

        y.backward();

        assert(x.grads[0, 0].approxEqual(1.0 / 1.0));
        assert(x.grads[0, 1].approxEqual(1.0 / 2.0));
        assert(x.grads[1, 0].approxEqual(1.0 / 3.0));
        assert(x.grads[1, 1].approxEqual(1.0 / 4.0));
    }
}

version (all) // sigmoid
{
    Tensor!(T, Shape, useGradient) sigmoid(T, size_t[] Shape, UseGradient useGradient)(Tensor!(T, Shape, useGradient) x)
    {
        import std.math : exp;

        auto y = x.value.map!(a => T(1) / (T(1) + exp(-a))).slice();

        alias Return = typeof(return);
        alias Value = Return.Value;

        static if (canBackward!(typeof(x)))
        {
            x.usedCount++;

            return new Tensor!(T, Shape)(y, (Value grads) {
                x.backward(y * (T(1) - y) * grads);
            });
        }
        else
        {
            return new Tensor!(T, Shape, No.gradient)(y);
        }
    }

    unittest
    {
        auto x = tensor!([3, 1])([-1.0f, 0.0f, 1.0f]);
        auto y = sigmoid(x);

        import std.format : format;
        import std.math : exp, approxEqual;

        assert(y.value[0, 0].approxEqual(1.0f / (1.0f + exp(+1.0f))), "%s".format(y.value));
        assert(y.value[1, 0].approxEqual(1.0f / (1.0f + exp(0.0f))), "%s".format(y.value));
        assert(y.value[2, 0].approxEqual(1.0f / (1.0f + exp(-1.0f))), "%s".format(y.value));

        y.backward();

        assert(x.grads[0, 0].approxEqual(y.value[0, 0] * (1.0 - y.value[0, 0])),
                "%s".format(x.grads));
        assert(x.grads[1, 0].approxEqual(y.value[1, 0] * (1.0 - y.value[1, 0])),
                "%s".format(x.grads));
        assert(x.grads[2, 0].approxEqual(y.value[2, 0] * (1.0 - y.value[2, 0])),
                "%s".format(x.grads));
    }

    unittest
    {
        auto x = tensor!([3, 1], No.gradient)([-1.0f, 0.0f, 1.0f]);
        auto y = sigmoid(x);
        
        import std.format : format;
        import std.math : exp, approxEqual;

        assert(y.value[0, 0].approxEqual(1.0f / (1.0f + exp(+1.0f))), "%s".format(y.value));
        assert(y.value[1, 0].approxEqual(1.0f / (1.0f + exp(0.0f))), "%s".format(y.value));
        assert(y.value[2, 0].approxEqual(1.0f / (1.0f + exp(-1.0f))), "%s".format(y.value));
    }
}

version (all) // tanh
{
    Tensor!(T, Shape, useGradient) tanh(T, size_t[] Shape, UseGradient useGradient)(Tensor!(T, Shape, useGradient) x)
    {
        import std.math : stdtanh = tanh;

        auto y = slice(x.value.map!(a => stdtanh(a)));

        static if (canBackward!(typeof(x)))
        {
            x.usedCount++;

            return new typeof(return)(y, (Slice!(T*, Shape.length) grads) {
                x.backward((1 - y * y) * grads);
            });
        }
        else
        {
            return new typeof(return)(y);
        }
    }

    unittest
    {
        auto x = tensor!([2])([-1.0f, 1.0f]);
        auto y = tanh(x);

        import std.math : stdtanh = tanh, approxEqual;

        assert(y.value[0].approxEqual(stdtanh(-1.0f)));
        assert(y.value[1].approxEqual(stdtanh(1.0f)));

        y.resetGrads();
        y.backward();

        import std : format;

        assert(x.grads[0].approxEqual(1 - y.value[0] ^^ 2),
                "%s : %s".format(x.grads[0], y.value[0]));
        assert(x.grads[1].approxEqual(1 - y.value[1] ^^ 2),
                "%s : %s".format(x.grads[1], y.value[1]));
    }
    
    unittest
    {
        auto x = tensor!([2], No.gradient)([-1.0f, 1.0f]);
        auto y = tanh(x);

        import std.math : stdtanh = tanh, approxEqual;

        assert(y.value[0].approxEqual(stdtanh(-1.0f)));
        assert(y.value[1].approxEqual(stdtanh(1.0f)));
    }
}

version (all) // sinh
{
    Tensor!(T, Shape, useGradient) sinh(T, size_t[] Shape, UseGradient useGradient)(Tensor!(T, Shape, useGradient) x)
    {
        import std.math : stdsinh = sinh, stdcosh = cosh;

        auto y = slice(x.value.map!(a => stdsinh(a)));

        static if (canBackward!(typeof(x)))
        {
            x.usedCount++;

            return new typeof(return)(y, (Slice!(T*, Shape.length) grads) {
                x.backward(x.value.map!stdcosh * grads);
            });
        }
        else
        {
            return new typeof(return)(y);
        }
    }

    unittest
    {
        auto x = tensor!([2])([-1.0f, 1.0f]);
        auto y = sinh(x);

        import std.math : stdsinh = sinh, stdcosh = cosh, approxEqual;

        assert(y.value[0].approxEqual(stdsinh(-1.0f)));
        assert(y.value[1].approxEqual(stdsinh(1.0f)));

        y.resetGrads();
        y.backward();

        import std : format;

        assert(x.grads[0].approxEqual(stdcosh(-1.0f)),
                "%s : %s".format(x.grads[0], y.value[0]));
        assert(x.grads[1].approxEqual(stdcosh(1.0f)),
                "%s : %s".format(x.grads[1], y.value[1]));
    }
    
    unittest
    {
        auto x = tensor!([2], No.gradient)([-1.0f, 1.0f]);
        auto y = sinh(x);

        import std.math : stdsinh = sinh, approxEqual;

        assert(y.value[0].approxEqual(stdsinh(-1.0f)));
        assert(y.value[1].approxEqual(stdsinh(1.0f)));
    }
}

version (all) // asinh
{
    Tensor!(T, Shape, useGradient) asinh(T, size_t[] Shape, UseGradient useGradient)(Tensor!(T, Shape, useGradient) x)
    {
        import std.math : stdasinh = asinh, stdsqrt = sqrt;

        auto y = slice(x.value.map!(a => stdasinh(a)));

        static if (canBackward!(typeof(x)))
        {
            x.usedCount++;

            return new typeof(return)(y, (Slice!(T*, Shape.length) grads) {
                x.backward(x.value.map!(a => T(1) / stdsqrt(a * a + T(1))) * grads);
            });
        }
        else
        {
            return new typeof(return)(y);
        }
    }

    unittest
    {
        auto x = tensor!([2])([-1.0f, 1.0f]);
        auto y = asinh(x);

        import std.math : stdasinh = asinh, stdsqrt = sqrt, approxEqual;

        assert(y.value[0].approxEqual(stdasinh(-1.0f)));
        assert(y.value[1].approxEqual(stdasinh(1.0f)));

        y.resetGrads();
        y.backward();

        import std : format;

        assert(x.grads[0].approxEqual(1 / stdsqrt(-1.0f * -1.0f + 1)),
                "%s : %s".format(x.grads[0], y.value[0]));
        assert(x.grads[1].approxEqual(1 / stdsqrt(1.0f * 1.0f + 1)),
                "%s : %s".format(x.grads[1], y.value[1]));
    }
    
    unittest
    {
        auto x = tensor!([2], No.gradient)([-1.0f, 1.0f]);
        auto y = asinh(x);

        import std.math : stdasinh = asinh, approxEqual;

        assert(y.value[0].approxEqual(stdasinh(-1.0f)));
        assert(y.value[1].approxEqual(stdasinh(1.0f)));
    }
}

version (all) // cosh
{
    Tensor!(T, Shape, useGradient) cosh(T, size_t[] Shape, UseGradient useGradient)(Tensor!(T, Shape, useGradient) x)
    {
        import std.math : stdcosh = cosh, stdsinh = sinh;

        auto y = slice(x.value.map!(a => stdcosh(a)));

        static if (canBackward!(typeof(x)))
        {
            x.usedCount++;

            return new typeof(return)(y, (Slice!(T*, Shape.length) grads) {
                x.backward(x.value.map!stdsinh * grads);
            });
        }
        else
        {
            return new typeof(return)(y);
        }
    }

    unittest
    {
        auto x = tensor!([2])([-1.0f, 1.0f]);
        auto y = cosh(x);

        import std.math : stdcosh = cosh, stdsinh = sinh, approxEqual;

        assert(y.value[0].approxEqual(stdcosh(-1.0f)));
        assert(y.value[1].approxEqual(stdcosh(1.0f)));

        y.resetGrads();
        y.backward();

        import std : format;

        assert(x.grads[0].approxEqual(stdsinh(-1.0f)),
                "%s : %s".format(x.grads[0], y.value[0]));
        assert(x.grads[1].approxEqual(stdsinh(1.0f)),
                "%s : %s".format(x.grads[1], y.value[1]));
    }
    
    unittest
    {
        auto x = tensor!([2], No.gradient)([-1.0f, 1.0f]);
        auto y = cosh(x);

        import std.math : stdcosh = cosh, approxEqual;

        assert(y.value[0].approxEqual(stdcosh(-1.0f)));
        assert(y.value[1].approxEqual(stdcosh(1.0f)));
    }
}

version (all) // acosh
{
    Tensor!(T, Shape, useGradient) acosh(T, size_t[] Shape, UseGradient useGradient)(Tensor!(T, Shape, useGradient) x)
    {
        import std.math : stdacosh = acosh, stdsqrt = sqrt;

        auto y = slice(x.value.map!(a => stdacosh(a)));

        static if (canBackward!(typeof(x)))
        {
            x.usedCount++;

            return new typeof(return)(y, (Slice!(T*, Shape.length) grads) {
                x.backward(x.value.map!(a => 1 / (stdsqrt(a - 1) * stdsqrt(a + 1))) * grads);
            });
        }
        else
        {
            return new typeof(return)(y);
        }
    }

    unittest
    {
        auto x = tensor!([2])([2.0f, 3.0f]);
        auto y = acosh(x);

        import std.math : stdacosh = acosh, stdsqrt = sqrt, approxEqual;

        assert(y.value[0].approxEqual(stdacosh(2.0f)));
        assert(y.value[1].approxEqual(stdacosh(3.0f)));

        y.resetGrads();
        y.backward();

        import std : format;

        assert(x.grads[0].approxEqual(1 / (stdsqrt(2.0f - 1) * stdsqrt(2.0f + 1))),
                "%s : %s".format(x.grads[0], y.value[0]));
        assert(x.grads[1].approxEqual(1 / (stdsqrt(3.0f - 1) * stdsqrt(3.0f + 1))),
                "%s : %s".format(x.grads[1], y.value[1]));
    }
    
    unittest
    {
        auto x = tensor!([2], No.gradient)([2.0f, 3.0f]);
        auto y = acosh(x);

        import std.math : stdacosh = acosh, approxEqual;

        assert(y.value[0].approxEqual(stdacosh(2.0f)));
        assert(y.value[1].approxEqual(stdacosh(3.0f)));
    }
}

version (all) // softplus
{
    Tensor!(T, Shape, useGradient) softplus(T, size_t[] Shape, UseGradient useGradient)(Tensor!(T, Shape, useGradient) x)
    {
        import std.math : stdlog = log, stdexp = exp;

        auto y = slice(x.value.map!(a => T(stdlog(1 + stdexp(a)))));

        static if (canBackward!(typeof(x)))
        {
            x.usedCount++;

            return new typeof(return)(y, (Slice!(T*, Shape.length) grads) {
                x.backward(x.value.map!(a => (stdexp(a) / (stdexp(a) + 1))) * grads);
            });
        }
        else
        {
            return new typeof(return)(y);
        }
    }

    unittest
    {
        auto x = tensor!([2])([-1.0f, 1.0f]);
        auto y = softplus(x);

        import std.math : stdlog = log, stdexp = exp, approxEqual;

        assert(y.value[0].approxEqual(stdlog(1 + stdexp(-1.0f))));
        assert(y.value[1].approxEqual(stdlog(1 + stdexp(1.0f))));

        y.resetGrads();
        y.backward();

        import std : format;

        assert(x.grads[0].approxEqual(stdexp(-1.0f) / (stdexp(-1.0f) + 1)),
                "%s : %s".format(x.grads[0], y.value[0]));
        assert(x.grads[1].approxEqual(stdexp(1.0f) / (stdexp(1.0f) + 1)),
                "%s : %s".format(x.grads[1], y.value[1]));
    }
    
    unittest
    {
        auto x = tensor!([2], No.gradient)([-1.0f, 1.0f]);
        auto y = softplus(x);

        import std.math : stdlog = log, stdexp = exp, approxEqual;

        assert(y.value[0].approxEqual(stdlog(1 + stdexp(-1.0f))));
        assert(y.value[1].approxEqual(stdlog(1 + stdexp(1.0f))));
    }
}

version (all) // relu
{
    Tensor!(T, Shape, useGradient) relu(T, size_t[] Shape, UseGradient useGradient)(Tensor!(T, Shape, useGradient) x)
    {
        import std.algorithm : max;

        auto y = slice(x.value.map!(a => max(T(0), a)));

        alias Return = typeof(return);
        alias Value = Return.Value;

        static if (canBackward!(typeof(x)))
        {
            x.usedCount++;

            return new typeof(return)(y, (Value grad) {
                x.backward(grad * x.value.map!(a => T(a > 0 ? 1 : 0)));
            });
        }
        else
        {
            return new typeof(return)(y);
        }
    }

    unittest
    {
        auto x = tensor!([2, 3])([-1.0, 0.0, 1.0, -2.0, 0.0, 2.0]);

        auto y = relu(x);

        y.backward();

        assert(y.value[0, 0] == 0);
        assert(y.value[0, 1] == 0);
        assert(y.value[0, 2] == 1.0);
        assert(y.value[1, 0] == 0);
        assert(y.value[1, 1] == 0);
        assert(y.value[1, 2] == 2.0);

        assert(x.grads[0, 0] == 0);
        assert(x.grads[0, 1] == 0);
        assert(x.grads[0, 2] == 1.0);
        assert(x.grads[1, 0] == 0);
        assert(x.grads[1, 1] == 0);
        assert(x.grads[1, 2] == 1.0);
    }
    
    unittest
    {
        auto x = tensor!([2, 3], No.gradient)([-1.0, 0.0, 1.0, -2.0, 0.0, 2.0]);
        auto y = relu(x);

        assert(y.value[0, 0] == 0);
        assert(y.value[0, 1] == 0);
        assert(y.value[0, 2] == 1.0);
        assert(y.value[1, 0] == 0);
        assert(y.value[1, 1] == 0);
        assert(y.value[1, 2] == 2.0);
    }
}

version (all) // linear
{
    // dfmt off
    Tensor!(T, [ShapeX[0], ShapeW[1]], useGradX | useGradW | useGradB) linear(
        T,
        size_t[2] ShapeX, UseGradient useGradX,
        size_t[2] ShapeW, UseGradient useGradW,
        size_t[1] ShapeB, UseGradient useGradB
    )(
        Tensor!(T, ShapeX, useGradX) x,
        Tensor!(T, ShapeW, useGradW) W,
        Tensor!(T, ShapeB, useGradB) B
    )
    // dfmt on
    {
        static assert(ShapeX[1] == ShapeW[0]);
        static assert(ShapeW[1] == ShapeB[0]);

        enum OutputDim = ShapeW[1];

        auto batchSize = x.value.shape[0];
        auto result = uninitSlice!T([batchSize, OutputDim]);
        foreach (i; 0 .. result.shape[0])
        {
            result[i, 0 .. $] = B.value[];
        }

        import mir.blas : gemm;

        gemm(T(1), x.value, W.value, T(1), result);

        alias Return = typeof(return);
        alias Value = Return.Value;

        static if (useGradW | useGradX | useGradB)
        {
            static if (canBackward!(typeof(W))) W.usedCount++;
            static if (canBackward!(typeof(x))) x.usedCount++;
            static if (canBackward!(typeof(B))) B.usedCount++;

            return new Return(result, (Value grad) {
                static if (canBackward!(typeof(W))) 
                {
                    W.backward((ref wGrads) {
                        gemm(T(1), x.value.transposed, grad, T(1), wGrads);
                    });
                }
                static if (canBackward!(typeof(x))) 
                {
                    x.backward((ref xGrads) {
                        gemm(T(1), grad, W.value.transposed, T(1), xGrads);
                    });
                }
                static if (canBackward!(typeof(B))) 
                {
                    B.backward((ref bGrads) {
                        foreach (i; 0 .. grad.shape[0])
                        {
                            bGrads[] += grad[i, 0 .. $];
                        }
                    });
                }
            });
        }
        else
        {
            return new Return(result);
        }
    }

    unittest
    {
        // static
        Tensor!(float, [2, 2]) w = tensor!([2, 2])([1.0f, 2.0f, 3.0f, 4.0f]);
        Tensor!(float, [2]) b = tensor!([2])([100.0f, 200.0f]);
        // dynamic batchSize
        Tensor!(float, [0, 2]) x = tensor!([0, 2])([1.0f, 2.0f]);

        // result
        Tensor!(float, [0, 2]) z = linear(x, w, b);

        assert(z.value[0, 0] == 1 * 1 + 2 * 3 + 100);
        assert(z.value[0, 1] == 1 * 2 + 2 * 4 + 200);

        z.backward();
    }

    unittest
    {
        // miss input dim
        Tensor!(float, [15, 3]) weights;
        Tensor!(float, [3]) bias;

        Tensor!(float, [0, 4, 4]) x;

        // compile error
        static assert(!__traits(compiles, {
                auto y = x.flatten().linear(weights, bias);
            }));
    }

    unittest
    {
        Tensor!(float, [16, 3]) weights;
        Tensor!(float, [3]) bias;

        Tensor!(float, [0, 4, 4]) x;

        static assert(__traits(compiles, {
                auto y = x.flatten().linear(weights, bias);
            }));
    }

    unittest
    {
        auto w = tensor!([2, 3])([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]);
        auto x = tensor!([0, 2])([1.0f, 2.0f, 3.0f, 4.0f]);
        auto b = tensor!([3])([100.0f, 200.0f, 300.0f]);

        auto z = linear(x, w, b);

        assert(z.value[0] == [1 * 1 + 2 * 4 + 100, 1 * 2 + 2 * 5 + 200, 1 * 3 + 2 * 6 + 300]);
        assert(z.value[1] == [3 * 1 + 4 * 4 + 100, 3 * 2 + 4 * 5 + 200, 3 * 3 + 4 * 6 + 300]);
    }
    
    unittest
    {
        auto w = tensor!([2, 3], No.gradient)([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]);
        auto x = tensor!([0, 2])([1.0f, 2.0f, 3.0f, 4.0f]);
        auto b = tensor!([3])([100.0f, 200.0f, 300.0f]);

        auto z = linear(x, w, b);

        assert(z.value[0] == [1 * 1 + 2 * 4 + 100, 1 * 2 + 2 * 5 + 200, 1 * 3 + 2 * 6 + 300]);
        assert(z.value[1] == [3 * 1 + 4 * 4 + 100, 3 * 2 + 4 * 5 + 200, 3 * 3 + 4 * 6 + 300]);
    }
    
    unittest
    {
        auto w = tensor!([2, 3])([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]);
        auto x = tensor!([0, 2], No.gradient)([1.0f, 2.0f, 3.0f, 4.0f]);
        auto b = tensor!([3])([100.0f, 200.0f, 300.0f]);

        auto z = linear(x, w, b);

        assert(z.value[0] == [1 * 1 + 2 * 4 + 100, 1 * 2 + 2 * 5 + 200, 1 * 3 + 2 * 6 + 300]);
        assert(z.value[1] == [3 * 1 + 4 * 4 + 100, 3 * 2 + 4 * 5 + 200, 3 * 3 + 4 * 6 + 300]);
    }
    
    unittest
    {
        auto w = tensor!([2, 3])([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]);
        auto x = tensor!([0, 2])([1.0f, 2.0f, 3.0f, 4.0f]);
        auto b = tensor!([3], No.gradient)([100.0f, 200.0f, 300.0f]);

        auto z = linear(x, w, b);

        assert(z.value[0] == [1 * 1 + 2 * 4 + 100, 1 * 2 + 2 * 5 + 200, 1 * 3 + 2 * 6 + 300]);
        assert(z.value[1] == [3 * 1 + 4 * 4 + 100, 3 * 2 + 4 * 5 + 200, 3 * 3 + 4 * 6 + 300]);
    }

    unittest
    {
        auto w = tensor!([2, 3], No.gradient)([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]);
        auto x = tensor!([0, 2], No.gradient)([1.0f, 2.0f, 3.0f, 4.0f]);
        auto b = tensor!([3], No.gradient)([100.0f, 200.0f, 300.0f]);
        
        auto z = linear(x, w, b);
        static assert(!canBackward!(typeof(z)));

        assert(z.value[0] == [1 * 1 + 2 * 4 + 100, 1 * 2 + 2 * 5 + 200, 1 * 3 + 2 * 6 + 300]);
        assert(z.value[1] == [3 * 1 + 4 * 4 + 100, 3 * 2 + 4 * 5 + 200, 3 * 3 + 4 * 6 + 300]);
    }
}

version (all) // sum
{
    Tensor!(T, [1], useGradient) sum(alias mode = "fast", T, size_t[] Shape, UseGradient useGradient)(Tensor!(T, Shape, useGradient) x)
            if (Shape.length == 2 && Shape[1] == 1)
    {
        import mir.math.sum : mirsum = sum;

        auto y = slice!T([1], mirsum!mode(x.value));

        alias Return = typeof(return);
        alias Value = Return.Value;

        static if (canBackward!(typeof(x)))
        {
            x.usedCount++;

            return new Return(y, (Value grad) {
                x.backward((ref xGrads) { xGrads[] += grad; });
            });
        }
        else
        {
            return new Return(y);
        }
    }

    unittest
    {
        auto x = tensor!([4, 1])([1.0f, 2.0f, 3.0f, 4.0f]);
        auto s = sum(x);
        assert(s.value[0] == 10.0f);

        assert(x.grads == [[0.0f], [0.0f], [0.0f], [0.0f]]);
        s.backward();
        assert(x.grads == [[1.0f], [1.0f], [1.0f], [1.0f]]);
    }
    
    unittest
    {
        auto x = tensor!([4, 1], No.gradient)([1.0f, 2.0f, 3.0f, 4.0f]);
        auto s = sum(x);
        assert(s.value[0] == 10.0f);
    }

    Tensor!(T, [Shape[0], 1], useGradient) sum(alias mode = "fast", T, size_t[] Shape, UseGradient useGradient)(Tensor!(T, Shape, useGradient) x)
            if ((Shape.length == 2 && Shape[1] != 1) || (Shape.length > 2))
    {
        import mir.math.sum : mirsum = sum;

        const batchSize = x.value.shape[0];
        auto y = uninitSlice!T([batchSize, 1]);
        foreach (i; 0 .. batchSize)
        {
            y[i, 0] = mirsum!mode(x.value[i]);
        }

        alias Return = typeof(return);
        alias Value = Return.Value;

        static if (canBackward!(typeof(x)))
        {
            x.usedCount++;

            return new Return(y, (Value grad) {
                x.backward((ref xGrads) {
                    foreach (i; 0 .. xGrads.shape[0])
                    {
                        xGrads[i].flattened[] = grad[i, 0];
                    }
                });
            });
        }
        else
        {
            return new Return(y);
        }
    }

    unittest
    {
        import std.format : format;

        auto x = tensor!([0, 4])([0.5, 1.0, 1.5, 2.0]);
        auto y = sum(x);

        assert(y.staticShape == [0, 1]);
        assert(y.value[0, 0] == 5.0);

        assert(x.grads == [[0, 0, 0, 0]], "%s".format(x.grads));
        y.backward();
        assert(x.grads == [[1, 1, 1, 1]], "%s".format(x.grads));
    }

    unittest
    {
        auto x = tensor!([2, 2, 2])([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        Tensor!(double, [2, 1]) y = sum(x);

        assert(y.value[0, 0] == 10.0);
        assert(y.value[1, 0] == 26.0);

        y.backward();
        assert(x.grads[0, 0, 0] == 1.0);
        assert(x.grads[0, 0, 1] == 1.0);
        assert(x.grads[0, 1, 0] == 1.0);
        assert(x.grads[0, 1, 1] == 1.0);
        assert(x.grads[1, 0, 0] == 1.0);
        assert(x.grads[1, 0, 1] == 1.0);
        assert(x.grads[1, 1, 0] == 1.0);
        assert(x.grads[1, 1, 1] == 1.0);
    }
    
    unittest
    {
        auto x = tensor!([2, 2, 2], No.gradient)([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        Tensor!(double, [2, 1], No.gradient) y = sum(x);

        assert(y.value[0, 0] == 10.0);
        assert(y.value[1, 0] == 26.0);
    }
}

version (all) // mean
{
    Tensor!(T, [1], useGradient) mean(alias mode = "fast", T, size_t[] Shape, UseGradient useGradient)(Tensor!(T, Shape, useGradient) x)
            if (Shape.length == 2 && Shape[1] == 1)
    {
        import mir.math.sum : mirsum = sum;

        const n = elementSize(x.value.shape);
        auto y = slice!T([1], mirsum!mode(x.value) / n);

        alias Return = typeof(return);
        alias Value = Return.Value;

        static if (canBackward!(typeof(x)))
        {
            x.usedCount++;

            return new Return(y, (Value grad) {
                x.backward((ref xGrads) { xGrads[] += grad / n; });
            });
        }
        else
        {
            return new Return(y);
        }
    }

    unittest
    {
        auto x = tensor!([4, 1])([1.0f, 2.0f, 3.0f, 4.0f]);
        auto s = mean(x);
        assert(s.value[0] == 2.5f);

        assert(x.grads == [[0.0f], [0.0f], [0.0f], [0.0f]]);
        s.backward();
        assert(x.grads == [[0.25f], [0.25f], [0.25f], [0.25f]]);
    }
    
    unittest
    {
        auto x = tensor!([4, 1], No.gradient)([1.0f, 2.0f, 3.0f, 4.0f]);
        auto s = mean(x);
        assert(s.value[0] == 2.5f);
    }

    Tensor!(T, [Shape[0], 1], useGradient) mean(alias mode = "fast", T, size_t[] Shape, UseGradient useGradient)(Tensor!(T, Shape, useGradient) x)
            if ((Shape.length == 2 && Shape[1] != 1) || (Shape.length > 2))
    {
        import mir.math.sum : mirsum = sum;

        const batchSize = x.value.shape[0];
        const n = elementSize(x.value.shape[1 .. $]);
        auto y = uninitSlice!T([batchSize, 1]);
        foreach (i; 0 .. batchSize)
        {
            y[i, 0] = mirsum!mode(x.value[i]) / n;
        }

        alias Return = typeof(return);
        alias Value = Return.Value;

        static if (canBackward!(typeof(x)))
        {
            x.usedCount++;

            return new Return(y, (Value grad) {
                x.backward((ref xGrads) {
                    foreach (i; 0 .. xGrads.shape[0])
                    {
                        xGrads[i].flattened[] = grad[i, 0] / n;
                    }
                });
            });
        }
        else
        {
            return new Return(y);
        }
    }

    unittest
    {
        import std.format : format;

        auto x = tensor!([0, 4])([0.5, 1.0, 1.5, 2.0]);
        auto y = mean(x);

        assert(y.staticShape == [0, 1]);
        assert(y.value[0, 0] == 1.25);

        assert(x.grads == [[0, 0, 0, 0]], "%s".format(x.grads));
        y.backward();
        assert(x.grads == [[0.25, 0.25, 0.25, 0.25]], "%s".format(x.grads));
    }

    unittest
    {
        auto x = tensor!([2, 2, 2])([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        Tensor!(double, [2, 1]) y = mean(x);

        assert(y.value[0, 0] == 2.5);
        assert(y.value[1, 0] == 6.5);

        y.backward();
        assert(x.grads[0, 0, 0] == 0.25);
        assert(x.grads[0, 0, 1] == 0.25);
        assert(x.grads[0, 1, 0] == 0.25);
        assert(x.grads[0, 1, 1] == 0.25);
        assert(x.grads[1, 0, 0] == 0.25);
        assert(x.grads[1, 0, 1] == 0.25);
        assert(x.grads[1, 1, 0] == 0.25);
        assert(x.grads[1, 1, 1] == 0.25);
    }
    
    unittest
    {
        auto x = tensor!([2, 2, 2], No.gradient)([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        Tensor!(double, [2, 1], No.gradient) y = mean(x);

        assert(y.value[0, 0] == 2.5);
        assert(y.value[1, 0] == 6.5);
    }
}

version (all) // flatten
{
    Tensor!(T, [Shape[0], elementSize(Shape[1 .. $])], useGradient) flatten(T, size_t[] Shape, UseGradient useGradient)(Tensor!(T, Shape, useGradient) x)
    {
        int err;
        auto y = x.value.reshape([x.value.shape[0], -1], err);
        assert(err == 0);

        static if (canBackward!(typeof(x)))
        {
            x.usedCount++;

            alias Value = typeof(return).Value;
            return new typeof(return)(y, (Value grad) {
                int err;
                auto reshaped = grad.reshape([
                        grad.shape[0], expandShape!(Shape[1 .. $])
                    ], err);
                assert(err == 0);
                x.backward(reshaped);
            });
        }
        else
        {
            return new typeof(return)(y);
        }
    }

    unittest
    {
        auto x = tensor!([2, 2, 2])([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        Tensor!(double, [2, 4]) y = flatten(x);
        assert(y.staticShape == [2, 4]);
        assert(y.value[0, 0] == 1.0);
        assert(y.value[0, 1] == 2.0);
        assert(y.value[0, 2] == 3.0);
        assert(y.value[0, 3] == 4.0);
        assert(y.value[1, 0] == 5.0);
        assert(y.value[1, 1] == 6.0);
        assert(y.value[1, 2] == 7.0);
        assert(y.value[1, 3] == 8.0);
    }
    
    unittest
    {
        auto x = tensor!([2, 2, 2], No.gradient)([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        Tensor!(double, [2, 4], No.gradient) y = flatten(x);
    }
}

version (all) // softmax
{
    Tensor!(T, Shape, useGrad) softmax(T, size_t[] Shape, UseGradient useGrad)(Tensor!(T, Shape, useGrad) x)
        if (Shape.length == 2)
    {
        import std.math : stdexp = exp;

        static if (Shape[0] == 0)
            const batchSize = x.shape[0];
        else
            enum batchSize = Shape[0];

        enum dim = Shape[1];
        auto y = uninitSlice!T(batchSize, dim);
        
        const expx = slice(x.value.map!stdexp);
        T[dim] temp;
        foreach (i; 0 .. batchSize)
        {
            auto s = T(0);
            foreach (j; 0 .. dim)
            {
                temp[j] = expx[i, j];
                s += temp[j];
            }
            foreach (j; 0 .. dim)
            {
                y[i, j] = temp[j] / s;
            }
        }

        static if (useGrad)
        {
            x.usedCount++;
            return new Tensor!(T, Shape)(y, (grads) {
                x.backward((ref xGrads) {
                    foreach (i; 0 .. batchSize)
                    {
                        import mir.math.sum : mirsum = sum;

                        const s = mirsum!"fast"(expx[i, 0 .. dim]);
                        const is2 = T(1) / (s * s);
                        foreach (j; 0 .. dim)
                        {
                            const a = grads[i, j];
                            auto d = T(0);
                            foreach (k; 0 .. dim)
                            {
                                if (k == j) continue;
                                d += (a - grads[i, k]) * expx[i, k];
                            }
                            
                            xGrads[i, j] = is2 * expx[i, j] * d;
                        }
                    }
                });
            });
        }
        else
        {
            return new Tensor!(T, Shape, UseGradient.no)(y);
        }
    }

    unittest
    {
        auto x = tensor!([0, 3])([[1.0, 2.0, 3.0]]);
        auto y = softmax(x);
        auto z = tensor!([0, 3], UseGradient.no)([[1.0, 0.0, 0.0]]);

        import mir.math.sum : mirsum = sum;
        import std.math : approxEqual;

        assert(mirsum(y.value).approxEqual(1));

        auto t = z - y;
        auto loss = mean(t * t);

        loss.backward();
    }
}

version (all) // softmaxCrossEntropy
{
    Tensor!(T, [1]) softmaxCrossEntropy(T, size_t[] Shape1, size_t[] Shape2, UseGradient useGrad)(Tensor!(T, Shape1, useGrad) x, Tensor!(T, Shape2, UseGradient.no) y)
    if (Shape1.length == 2 && Shape2.length == 2 && Shape1[1] == Shape2[1])
    {
        static assert(Shape1[0] == 0 || Shape2[0] == 0 || Shape1[0] == Shape2[0]);
        assert(x.shape[0] == y.shape[0]);

        import mir.ndslice : map, zip;
        import mir.math.sum : sum;
        import std.math : exp, log;

        const batchSize = x.shape[0];
        T[] u = new T[batchSize];
        T a = T(0);
        T b = T(0);
        foreach (i; 0 .. batchSize)
        {
            a += zip(x.value[i], y.value[i]).map!"a*b"().sum!"fast"();
            const temp = x.value[i].map!exp.sum!"fast"();
            u[i] = 1 / temp;
            b += log(temp);
        }

        auto z = slice([1], (b - a) / batchSize);

        x.usedCount++;
        alias Return = typeof(return);
        alias Value = Return.Value;

        return new Return(z, (Value grads) {
            x.backward((ref xGrads) {
                const c = grads[0] / batchSize;
                foreach (i; 0 .. batchSize)
                {
                    const ut = u[i];
                    xGrads[i][] += (x.value[i].map!(a => exp(a) * ut) - y.value[i]) * c;
                }
            });
        });
    }

    unittest
    {
        auto x = tensor!([0, 3])([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]);
        auto y = tensor!([0, 3], UseGradient.no)([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]);

        auto loss = softmaxCrossEntropy(x, y);
        static assert(loss.shape == [1]);

        import std.math : approxEqual, E;

        assert(loss.value[0].approxEqual(0.688236607616066));

        loss.backward();

        enum double g1 = 1.0 / (8.0 + 4 * E);
        enum double g2 = -1.0 / (4.0 + 2 * E);

        assert(x.grads[0, 0].approxEqual(1.0 / 12));
        assert(x.grads[0, 1].approxEqual(1.0 / 12));
        assert(x.grads[0, 2].approxEqual(-1.0 / 6));
        assert(x.grads[1, 0].approxEqual(g1));
        assert(x.grads[1, 1].approxEqual(g1));
        assert(x.grads[1, 2].approxEqual(g2));
        assert(x.grads[2, 0].approxEqual(g1));
        assert(x.grads[2, 1].approxEqual(g2));
        assert(x.grads[2, 2].approxEqual(g1));
        assert(x.grads[3, 0].approxEqual(g2));
        assert(x.grads[3, 1].approxEqual(g1));
        assert(x.grads[3, 2].approxEqual(g1));
    }
}

version (all) // dropout
{
    Tensor!(T, Shape, useGradient) dropout(T, size_t[] Shape, UseGradient useGradient)(Tensor!(T, Shape, useGradient) x, float rate, bool isTrain)
    {
        import std : roundTo;
        import golem.util : elementSize;
        import mir.ndslice : flattened;

        enum size = elementSize(Shape[1 .. $]);
        const dropSize = roundTo!size_t(size * (1 - rate));

        if (isTrain)
        {
            auto filter = onesLike(x);
            foreach (i; 0 .. x.shape[0])
            {
                import std.random : uniform;

                auto row = filter.value[i][].flattened;
                foreach (j; 0 .. dropSize)
                {
                    row[uniform(0, size)] = 0;
                }
            }
            return filter * x;
        }
        else
        {
            const p = T(size - dropSize) / size;
            return p * x;
        }
    }

    unittest
    {
        auto x = tensor!([0, 2, 2])([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        auto y = dropout(x, 0.5, true);
        auto z = dropout(x, 0.5, false);

        import std.algorithm : count;

        const t = y.value.flattened[].count(0);
        assert(t >= 2); // batchSize * 1
        assert(t <= 2 * 2); // batchSize * round(4 * 0.5)

        import std.math : round, approxEqual;

        const a = (4 - round(4 * 0.5)) / 4;
        assert(z.value[0, 0, 0].approxEqual(1.0 * a));
        assert(z.value[0, 0, 1].approxEqual(2.0 * a));
        assert(z.value[0, 1, 0].approxEqual(3.0 * a));
        assert(z.value[0, 1, 1].approxEqual(4.0 * a));
        assert(z.value[1, 0, 0].approxEqual(5.0 * a));
        assert(z.value[1, 0, 1].approxEqual(6.0 * a));
        assert(z.value[1, 1, 0].approxEqual(7.0 * a));
        assert(z.value[1, 1, 1].approxEqual(8.0 * a));
    }
}

version (all) // concat
{
    size_t[] makeConcatShape(size_t[] lhs, size_t[] rhs)
    in (lhs.length > 0)
    in (rhs.length > 0)
    in (lhs.length == rhs.length)
    {
        size_t axis = lhs.length - 1;
        foreach (i; 0 .. lhs.length)
        {
            if (lhs[i] != rhs[i])
            {
                axis = i;
                break;
            }
        }
        auto shape = lhs.dup;
        shape[axis] += rhs[axis];
        return shape;
    }

    template ConcatTensor(TensorL, TensorR)
    if (isTensor!TensorL && isTensor!TensorR)
    {
        import std.format : format;

        // dfmt off
        static assert(TensorL.staticShape.length == TensorR.staticShape.length,
            format!"%s != %s"(TensorL.staticShape, TensorR.staticShape));
        // dfmt on

        private alias ElementType = TensorL.ElementType;
        private enum Shape = makeConcatShape(TensorL.staticShape, TensorR.staticShape);
        private enum useGradient = commonGradientType!(TensorL, TensorR);

        alias ConcatTensor = Tensor!(ElementType, Shape, useGradient);
    }

    // Dim: [N, A] + [N, B] => [N, A + B]
    auto concat(T, U)(T x, U y)
    if (isTensor!T && isTensor!U)
    {
        import std.format : format;
        static assert(T.staticShape.length == 2, format!"Only 2 dimensions are supported at x (%s)"(T.staticShape));
        static assert(U.staticShape.length == 2, format!"Only 2 dimensions are supported at y (%s)"(U.staticShape));
        static if (T.staticShape[0] != 0 && U.staticShape[0] != 0)
        {
            static assert(T.staticShape[0] == U.staticShape[0], format!"mismatch batch size (%s != %s)"(T.staticShape, U.staticShape));
        }
        else
        {
            assert(x.shape[0] == y.shape[0], format!"mismatch batch size (%s != %s)"(T.staticShape, U.staticShape));
        }

        alias Return = ConcatTensor!(T, U);

        const batchSize = x.shape[0];
        auto z = uninitSlice!(T.ElementType)(batchSize, x.shape[1] + y.shape[1]);
        foreach (i; 0 .. batchSize)
        {
            z[i][0 .. x.shape[1]] = x.value[i][0 .. $];
            z[i][x.shape[1] .. $] = y.value[i][0 .. $];
        }

        static if (canBackward!(Return))
        {
            static if (canBackward!T) x.usedCount++;
            static if (canBackward!U) y.usedCount++;
            return new Return(z, (grads) {
                static if (canBackward!T)
                {
                    x.backward((ref xGrads) {
                        xGrads[] += grads[0 .. $, 0 .. x.shape[1]];
                    });
                }
                static if (canBackward!U)
                {
                    y.backward((ref yGrads) {
                        yGrads[] += grads[0 .. $, x.shape[1] .. $];
                    });
                }
            });
        }
        else
        {
            return new Return(z);
        }
    }

    unittest
    {
        auto x = tensor!([2, 2])([1.0f, 2.0f, 3.0f, 4.0f]);
        auto y = tensor!([2, 1])([10.0f, 20.0f]);

        auto z = concat(x, y);

        assert(z.value[0, 0] == 1.0f);
        assert(z.value[0, 1] == 2.0f);
        assert(z.value[0, 2] == 10.0f);
        assert(z.value[1, 0] == 3.0f);
        assert(z.value[1, 1] == 4.0f);
        assert(z.value[1, 2] == 20.0f);

        auto a = tensor!([2, 3])([[1.0f, 2.0f, 3.0f], [4.0f, 5.0f, 6.0f]]);
        (a * z).backward();

        import std.conv : to;
        assert(x.grads[0, 0] == 1.0f, x.grads.to!string());
        assert(x.grads[0, 1] == 2.0f, x.grads.to!string());
        assert(x.grads[1, 0] == 4.0f, x.grads.to!string());
        assert(x.grads[1, 1] == 5.0f, x.grads.to!string());
        assert(y.grads[0, 0] == 3.0f, y.grads.to!string());
        assert(y.grads[1, 0] == 6.0f, y.grads.to!string());
    }
    
    unittest
    {
        auto x = tensor!([2, 2], UseGradient.no)([1.0f, 2.0f, 3.0f, 4.0f]);
        auto y = tensor!([2, 1])([10.0f, 20.0f]);

        auto z = concat(x, y);

        assert(z.value[0, 0] == 1.0f);
        assert(z.value[0, 1] == 2.0f);
        assert(z.value[0, 2] == 10.0f);
        assert(z.value[1, 0] == 3.0f);
        assert(z.value[1, 1] == 4.0f);
        assert(z.value[1, 2] == 20.0f);

        auto a = tensor!([2, 3])([[1.0f, 2.0f, 3.0f], [4.0f, 5.0f, 6.0f]]);
        (a * z).backward();

        import std.conv : to;
        assert(y.grads[0, 0] == 3.0f, y.grads.to!string());
        assert(y.grads[1, 0] == 6.0f, y.grads.to!string());
    }
    
    unittest
    {
        auto x = tensor!([2, 2])([1.0f, 2.0f, 3.0f, 4.0f]);
        auto y = tensor!([2, 1], UseGradient.no)([10.0f, 20.0f]);

        auto z = concat(x, y);

        assert(z.value[0, 0] == 1.0f);
        assert(z.value[0, 1] == 2.0f);
        assert(z.value[0, 2] == 10.0f);
        assert(z.value[1, 0] == 3.0f);
        assert(z.value[1, 1] == 4.0f);
        assert(z.value[1, 2] == 20.0f);

        auto a = tensor!([2, 3])([[1.0f, 2.0f, 3.0f], [4.0f, 5.0f, 6.0f]]);
        (a * z).backward();

        import std.conv : to;
        assert(x.grads[0, 0] == 1.0f, x.grads.to!string());
        assert(x.grads[0, 1] == 2.0f, x.grads.to!string());
        assert(x.grads[1, 0] == 4.0f, x.grads.to!string());
        assert(x.grads[1, 1] == 5.0f, x.grads.to!string());
    }
    
    unittest
    {
        auto x = tensor!([2, 2], UseGradient.no)([1.0f, 2.0f, 3.0f, 4.0f]);
        auto y = tensor!([2, 1], UseGradient.no)([10.0f, 20.0f]);

        auto z = concat(x, y);
        static assert(!canBackward!(typeof(z)));
        
        assert(z.value[0, 0] == 1.0f);
        assert(z.value[0, 1] == 2.0f);
        assert(z.value[0, 2] == 10.0f);
        assert(z.value[1, 0] == 3.0f);
        assert(z.value[1, 1] == 4.0f);
        assert(z.value[1, 2] == 20.0f);
    }

    unittest
    {
        auto x = tensor!([1, 1])([10.0f, 20.0f, 30.0f]);
        auto y = tensor!([2, 2])([1.0f, 2.0f, 3.0f, 4.0f]);

        // mismatch batch size
        static assert(!__traits(compiles, concat(x, y)));
    }
    
    unittest
    {
        auto x = tensor!([0, 1])([10.0f, 20.0f, 30.0f]);
        auto y = tensor!([0, 2])([1.0f, 2.0f, 3.0f, 4.0f]);

        // mismatch batch size
        import core.exception : AssertError;
        import std.exception : assertThrown;

        assertThrown!AssertError(concat(x, y));
    }

    unittest
    {
        auto x = tensor!([3])([1.0f, 2.0f, 3.0f]);
        auto y = tensor!([3, 1])([1.0f, 2.0f, 3.0f]);
        auto z = tensor!([3, 1, 1])([1.0f, 2.0f, 3.0f]);
        
        static assert(!__traits(compiles, concat(x, y)));
        static assert(!__traits(compiles, concat(y, x)));
        static assert(!__traits(compiles, concat(y, z)));
        static assert(!__traits(compiles, concat(z, y)));
    }
}

version (all) // batchSum
{
    Tensor!(T, Shape[1 .. $], useGrad) batchSum(T, size_t[] Shape, UseGradient useGrad)(Tensor!(T, Shape, useGrad) x)
    {
        import mir.math.sum : mirsum = sum;

        auto y = x.value.bringToFront!(Shape.length - 1).pack!1.map!(a => mirsum(a)).slice();

        static if (useGrad)
        {
            x.usedCount++;
            return new typeof(return)(y, (grads) {
                x.backward((ref xGrads) {
                    foreach (i; 0 .. x.shape[0])
                    {
                        xGrads[i][] += grads[];
                    }
                });
            });
        }
        else
        {
            return new typeof(return)(y);
        }
    }

    unittest
    {
        auto x = tensor!([0, 2, 2])([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        
        Tensor!(double, [2, 2]) y = batchSum(x);
        assert(y.value == [[6.0, 8.0], [10.0, 12.0]]);

        y.backward();

        assert(x.grads == [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]);
    }

    unittest
    {
        auto x = tensor!([0, 2])([1.0, 2.0, 3.0, 4.0]);
        
        Tensor!(double, [2]) y = batchSum(x);
        assert(y.value == [4.0, 6.0]);

        auto z = y * tensor!([2])([-1.0, 2.0]);
        z.backward();

        assert(x.grads == [[-1.0, 2.0], [-1.0, 2.0]]);
    }

    unittest
    {
        auto x = tensor!([0, 2, 2], UseGradient.no)([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        
        Tensor!(double, [2, 2], UseGradient.no) y = batchSum(x);
        assert(y.value == [[8.0, 10.0], [12.0, 14.0]]);
    }
}
