module golem.math;

import golem.tensor;
import golem.util;

import mir.ndslice;

import std.typecons : No, tuple;

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

        import std.math : stdexp = exp, isClose;

        assert(y.value[0].isClose(stdexp(-1.0f)));
        assert(y.value[1].isClose(stdexp(1.0f)));

        y.resetGrads();
        y.backward();

        import std : format;

        assert(x.grads[0].isClose(y.value[0]), "%s : %s".format(x.grads[0], y.value[0]));
        assert(x.grads[1].isClose(y.value[1]), "%s : %s".format(x.grads[1], y.value[1]));
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
        
        import std.math : stdexp = exp, isClose;

        assert(y.value[0, 0].isClose(stdexp(1.0f)));
        assert(y.value[0, 1].isClose(stdexp(2.0f)));
        assert(y.value[1, 0].isClose(stdexp(3.0f)));
        assert(y.value[1, 1].isClose(stdexp(4.0f)));
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

        import std.math : stdlog = log, isClose;

        assert(y.value[0, 0].isClose(stdlog(1.0)));
        assert(y.value[0, 1].isClose(stdlog(2.0)));
        assert(y.value[1, 0].isClose(stdlog(3.0)));
        assert(y.value[1, 1].isClose(stdlog(4.0)));

        y.backward();

        assert(x.grads[0, 0].isClose(1.0 / 1.0));
        assert(x.grads[0, 1].isClose(1.0 / 2.0));
        assert(x.grads[1, 0].isClose(1.0 / 3.0));
        assert(x.grads[1, 1].isClose(1.0 / 4.0));
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
        import std.math : exp, isClose;

        assert(y.value[0, 0].isClose(1.0f / (1.0f + exp(+1.0f))), "%s".format(y.value));
        assert(y.value[1, 0].isClose(1.0f / (1.0f + exp(0.0f))), "%s".format(y.value));
        assert(y.value[2, 0].isClose(1.0f / (1.0f + exp(-1.0f))), "%s".format(y.value));

        y.backward();

        assert(x.grads[0, 0].isClose(y.value[0, 0] * (1.0 - y.value[0, 0])),
                "%s".format(x.grads));
        assert(x.grads[1, 0].isClose(y.value[1, 0] * (1.0 - y.value[1, 0])),
                "%s".format(x.grads));
        assert(x.grads[2, 0].isClose(y.value[2, 0] * (1.0 - y.value[2, 0])),
                "%s".format(x.grads));
    }

    unittest
    {
        auto x = tensor!([3, 1], No.gradient)([-1.0f, 0.0f, 1.0f]);
        auto y = sigmoid(x);
        
        import std.format : format;
        import std.math : exp, isClose;

        assert(y.value[0, 0].isClose(1.0f / (1.0f + exp(+1.0f))), "%s".format(y.value));
        assert(y.value[1, 0].isClose(1.0f / (1.0f + exp(0.0f))), "%s".format(y.value));
        assert(y.value[2, 0].isClose(1.0f / (1.0f + exp(-1.0f))), "%s".format(y.value));
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

        import std.math : stdtanh = tanh, isClose;

        assert(y.value[0].isClose(stdtanh(-1.0f)));
        assert(y.value[1].isClose(stdtanh(1.0f)));

        y.resetGrads();
        y.backward();

        import std : format;

        assert(x.grads[0].isClose(1 - y.value[0] ^^ 2),
                "%s : %s".format(x.grads[0], y.value[0]));
        assert(x.grads[1].isClose(1 - y.value[1] ^^ 2),
                "%s : %s".format(x.grads[1], y.value[1]));
    }
    
    unittest
    {
        auto x = tensor!([2], No.gradient)([-1.0f, 1.0f]);
        auto y = tanh(x);

        import std.math : stdtanh = tanh, isClose;

        assert(y.value[0].isClose(stdtanh(-1.0f)));
        assert(y.value[1].isClose(stdtanh(1.0f)));
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

        import std.math : stdsinh = sinh, stdcosh = cosh, isClose;

        assert(y.value[0].isClose(stdsinh(-1.0f)));
        assert(y.value[1].isClose(stdsinh(1.0f)));

        y.resetGrads();
        y.backward();

        import std : format;

        assert(x.grads[0].isClose(stdcosh(-1.0f)),
                "%s : %s".format(x.grads[0], y.value[0]));
        assert(x.grads[1].isClose(stdcosh(1.0f)),
                "%s : %s".format(x.grads[1], y.value[1]));
    }
    
    unittest
    {
        auto x = tensor!([2], No.gradient)([-1.0f, 1.0f]);
        auto y = sinh(x);

        import std.math : stdsinh = sinh, isClose;

        assert(y.value[0].isClose(stdsinh(-1.0f)));
        assert(y.value[1].isClose(stdsinh(1.0f)));
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

        import std.math : stdasinh = asinh, stdsqrt = sqrt, isClose;

        assert(y.value[0].isClose(stdasinh(-1.0f)));
        assert(y.value[1].isClose(stdasinh(1.0f)));

        y.resetGrads();
        y.backward();

        import std : format;

        assert(x.grads[0].isClose(1 / stdsqrt(-1.0f * -1.0f + 1)),
                "%s : %s".format(x.grads[0], y.value[0]));
        assert(x.grads[1].isClose(1 / stdsqrt(1.0f * 1.0f + 1)),
                "%s : %s".format(x.grads[1], y.value[1]));
    }
    
    unittest
    {
        auto x = tensor!([2], No.gradient)([-1.0f, 1.0f]);
        auto y = asinh(x);

        import std.math : stdasinh = asinh, isClose;

        assert(y.value[0].isClose(stdasinh(-1.0f)));
        assert(y.value[1].isClose(stdasinh(1.0f)));
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

        import std.math : stdcosh = cosh, stdsinh = sinh, isClose;

        assert(y.value[0].isClose(stdcosh(-1.0f)));
        assert(y.value[1].isClose(stdcosh(1.0f)));

        y.resetGrads();
        y.backward();

        import std : format;

        assert(x.grads[0].isClose(stdsinh(-1.0f)),
                "%s : %s".format(x.grads[0], y.value[0]));
        assert(x.grads[1].isClose(stdsinh(1.0f)),
                "%s : %s".format(x.grads[1], y.value[1]));
    }
    
    unittest
    {
        auto x = tensor!([2], No.gradient)([-1.0f, 1.0f]);
        auto y = cosh(x);

        import std.math : stdcosh = cosh, isClose;

        assert(y.value[0].isClose(stdcosh(-1.0f)));
        assert(y.value[1].isClose(stdcosh(1.0f)));
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

        import std.math : stdacosh = acosh, stdsqrt = sqrt, isClose;

        assert(y.value[0].isClose(stdacosh(2.0f)));
        assert(y.value[1].isClose(stdacosh(3.0f)));

        y.resetGrads();
        y.backward();

        import std : format;

        assert(x.grads[0].isClose(1 / (stdsqrt(2.0f - 1) * stdsqrt(2.0f + 1))),
                "%s : %s".format(x.grads[0], y.value[0]));
        assert(x.grads[1].isClose(1 / (stdsqrt(3.0f - 1) * stdsqrt(3.0f + 1))),
                "%s : %s".format(x.grads[1], y.value[1]));
    }
    
    unittest
    {
        auto x = tensor!([2], No.gradient)([2.0f, 3.0f]);
        auto y = acosh(x);

        import std.math : stdacosh = acosh, isClose;

        assert(y.value[0].isClose(stdacosh(2.0f)));
        assert(y.value[1].isClose(stdacosh(3.0f)));
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

        import std.math : stdlog = log, stdexp = exp, isClose;

        assert(y.value[0].isClose(stdlog(1 + stdexp(-1.0f))));
        assert(y.value[1].isClose(stdlog(1 + stdexp(1.0f))));

        y.resetGrads();
        y.backward();

        import std : format;

        assert(x.grads[0].isClose(stdexp(-1.0f) / (stdexp(-1.0f) + 1)),
                "%s : %s".format(x.grads[0], y.value[0]));
        assert(x.grads[1].isClose(stdexp(1.0f) / (stdexp(1.0f) + 1)),
                "%s : %s".format(x.grads[1], y.value[1]));
    }
    
    unittest
    {
        auto x = tensor!([2], No.gradient)([-1.0f, 1.0f]);
        auto y = softplus(x);

        import std.math : stdlog = log, stdexp = exp, isClose;

        assert(y.value[0].isClose(stdlog(1 + stdexp(-1.0f))));
        assert(y.value[1].isClose(stdlog(1 + stdexp(1.0f))));
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

version (all) // leakyRelu
{
    Tensor!(T, Shape, useGradient) leakyRelu(T, size_t[] Shape, UseGradient useGradient)(Tensor!(T, Shape, useGradient) x, T a = 0.01)
    {
        import std.algorithm : max;

        auto y = slice(x.value.map!(t => t > 0 ? t : a * t));

        alias Return = typeof(return);
        alias Value = Return.Value;

        static if (canBackward!(typeof(x)))
        {
            x.usedCount++;

            return new typeof(return)(y, (Value grad) {
                x.backward(grad * x.value.map!(t => T(t > 0 ? 1 : a)));
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

        auto y = leakyRelu(x, 0.02);

        y.backward();

        assert(y.value[0, 0] == -0.02);
        assert(y.value[0, 1] == 0);
        assert(y.value[0, 2] == 1.0);
        assert(y.value[1, 0] == -0.04);
        assert(y.value[1, 1] == 0);
        assert(y.value[1, 2] == 2.0);

        assert(x.grads[0, 0] == 0.02);
        assert(x.grads[0, 1] == 0.02);
        assert(x.grads[0, 2] == 1.0);
        assert(x.grads[1, 0] == 0.02);
        assert(x.grads[1, 1] == 0.02);
        assert(x.grads[1, 2] == 1.0);
    }
    
    unittest
    {
        auto x = tensor!([2, 3], No.gradient)([-1.0, 0.0, 1.0, -2.0, 0.0, 2.0]);
        auto y = leakyRelu(x, 0.2);

        assert(y.value[0, 0] == -0.2);
        assert(y.value[0, 1] == 0);
        assert(y.value[0, 2] == 1.0);
        assert(y.value[1, 0] == -0.4);
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

        const batchSize = x.value.shape[0];
        auto result = uninitSlice!T([batchSize, OutputDim]);
        foreach (i; 0 .. batchSize)
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
                        foreach (i; 0 .. batchSize)
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
        import std.math : isClose;

        assert(mirsum(y.value).isClose(1));

        auto t = z - y;
        auto loss = mean(t * t);

        loss.backward();
    }
}

version (all) // softmaxCrossEntropy
{
    Tensor!(T, [Shape1[0], 1], useGrad) softmaxCrossEntropy(T, size_t[] Shape1, size_t[] Shape2, UseGradient useGrad)(Tensor!(T, Shape1, useGrad) x, Tensor!(T, Shape2, UseGradient.no) y)
    if (Shape1.length == 2 && Shape2.length == 2 && Shape1[1] == Shape2[1])
    {
        static assert(Shape1[0] == 0 || Shape2[0] == 0 || Shape1[0] == Shape2[0]);
        assert(x.shape[0] == y.shape[0]);

        import mir.ndslice : map, zip;
        import mir.math.sum : sum;
        import std.math : exp, log;

        const c = T(1) / x.shape[1];

        auto t = x.value.ipack!1.map!(r => sum!"fast"(r.map!exp));

        int err;
        auto z = (c * (t.map!(a => cast(T) log(a)) - (x.value * y.value).ipack!1.map!(a => sum!"fast"(a))))
            .fuse()
            .reshape([x.shape[0], 1], err);

        static if (useGrad)
        {
            x.usedCount++;
        }
        alias Return = typeof(return);
        alias Value = Return.Value;

        static if (useGrad)
        {
            return new Return(z, (Value grads) {
                x.backward((ref xGrads) {
                    immutable p = T(1) / xGrads.shape[1];
                    foreach (i; 0 .. xGrads.shape[0])
                    {
                        xGrads[i][] += p * (x.value[i].map!exp / t[i] - y.value[i][]) * grads[i, 0];
                    }
                });
            });
        }
        else
        {
            return new Return(z);
        }
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
        assert(loss.shape == [4, 1]);

        import std.math : isClose, E;
        import std.format : format;

        assert(loss.value[0, 0].isClose(0.3662040962), format!"%.10f"(loss.value[0, 0]));
        assert(loss.value[1, 0].isClose(0.1838149046), format!"%.10f"(loss.value[1, 0]));
        assert(loss.value[2, 0].isClose(0.1838149046), format!"%.10f"(loss.value[2, 0]));
        assert(loss.value[3, 0].isClose(0.1838149046), format!"%.10f"(loss.value[3, 0]));

        loss.backward();

        enum double g1 = 1.0 / (6.0 + 3 * E);
        enum double g2 = -2.0 / (6.0 + 3 * E);

        import std.conv : text;

        assert(x.grads[0, 0].isClose(1.0 / 9), text(x.grads));
        assert(x.grads[0, 1].isClose(1.0 / 9), text(x.grads));
        assert(x.grads[0, 2].isClose(-2.0 / 9), text(x.grads));
        assert(x.grads[1, 0].isClose(g1), text(x.grads));
        assert(x.grads[1, 1].isClose(g1), text(x.grads));
        assert(x.grads[1, 2].isClose(g2), text(x.grads));
        assert(x.grads[2, 0].isClose(g1), text(x.grads));
        assert(x.grads[2, 1].isClose(g2), text(x.grads));
        assert(x.grads[2, 2].isClose(g1), text(x.grads));
        assert(x.grads[3, 0].isClose(g2), text(x.grads));
        assert(x.grads[3, 1].isClose(g1), text(x.grads));
        assert(x.grads[3, 2].isClose(g1), text(x.grads));
    }

    unittest
    {
        auto x = tensor!([0, 3], UseGradient.no)([
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
        assert(loss.shape == [4, 1]);

        import std.math : isClose, E;
        import std.format : format;

        assert(loss.value[0, 0].isClose(0.3662040962), format!"%.10f"(loss.value[0, 0]));
        assert(loss.value[1, 0].isClose(0.1838149046), format!"%.10f"(loss.value[1, 0]));
        assert(loss.value[2, 0].isClose(0.1838149046), format!"%.10f"(loss.value[2, 0]));
        assert(loss.value[3, 0].isClose(0.1838149046), format!"%.10f"(loss.value[3, 0]));

        static assert(!__traits(compiles, {
            loss.backward();
        }));
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

                auto row = filter.value[i].flattened;
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

        import std.math : round, isClose;

        const a = (4 - round(4 * 0.5)) / 4;
        assert(z.value[0, 0, 0].isClose(1.0 * a));
        assert(z.value[0, 0, 1].isClose(2.0 * a));
        assert(z.value[0, 1, 0].isClose(3.0 * a));
        assert(z.value[0, 1, 1].isClose(4.0 * a));
        assert(z.value[1, 0, 0].isClose(5.0 * a));
        assert(z.value[1, 0, 1].isClose(6.0 * a));
        assert(z.value[1, 1, 0].isClose(7.0 * a));
        assert(z.value[1, 1, 1].isClose(8.0 * a));
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
        auto z = uninitSlice!(T.ElementType)(batchSize, x.staticShape[1] + y.staticShape[1]);
        foreach (i; 0 .. batchSize)
        {
            z[i][0 .. x.staticShape[1]] = x.value[i][0 .. $];
            z[i][x.staticShape[1] .. $] = y.value[i][0 .. $];
        }

        static if (canBackward!(Return))
        {
            static if (canBackward!T) x.usedCount++;
            static if (canBackward!U) y.usedCount++;
            return new Return(z, (grads) {
                static if (canBackward!T)
                {
                    x.backward((ref xGrads) {
                        xGrads[] += grads[0 .. $, 0 .. x.staticShape[1]];
                    });
                }
                static if (canBackward!U)
                {
                    y.backward((ref yGrads) {
                        yGrads[] += grads[0 .. $, x.staticShape[1] .. $];
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

version (all) // boradcastOp
{
    template broadcastOp(string op)
    if (op == "+" || op == "-")
    {
        /+
        [N, C, W, H] + [C, W, H]
        [N, C, W, H] - [C, W, H]
        +/
        auto broadcastOp(T, size_t[] Shape1, UseGradient useGrad1, size_t[] Shape2, UseGradient useGrad2)(
                Tensor!(T, Shape1, useGrad1) x, Tensor!(T, Shape2, useGrad2) y)
            if (Shape1[$ - Shape2.length .. $] == Shape2)
        {
            enum Dim1 = Shape1.length;
            enum Dim2 = Shape2.length;

            static if (op == "+")
                alias binOp = (a, b) => a + b;
            else static if (op == "-")
                alias binOp = (a, b) => a - b;
            
            auto yv = y.value;
            auto z = x.value.pack!Dim2.map!(a => binOp(a, yv)).fuse();

            static if (useGrad1 || useGrad2)
            {
                static if (useGrad1) x.usedCount++;
                static if (useGrad2) y.usedCount++;
                return new Tensor!(T, Shape1, UseGradient.yes)(z, (grads) {
                    static if (useGrad1)
                    {
                        x.backward(grads);
                    }
                    static if (useGrad2)
                    {
                        y.backward((ref yGrads) {
                            import mir.math.sum : mirsum = sum;

                            static if (op == "+")
                                yGrads[] += grads.transposed!(expandIndex!(Dim1 - Dim2, Dim1)).ipack!Dim2.map!(a => mirsum(a));
                            else
                                yGrads[] -= grads.transposed!(expandIndex!(Dim1 - Dim2, Dim1)).ipack!Dim2.map!(a => mirsum(a));
                        });
                    }
                });
            }
            else
            {
                return new Tensor!(T, Shape1, UseGradient.no)(z);
            }
        }
    }

    unittest
    {
        auto x = tensor!([0, 2, 2], UseGradient.no)([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        auto y = tensor!([2], UseGradient.no)([10.0, 20.0]);

        auto z1 = broadcastOp!"+"(x, y);
        auto z2 = broadcastOp!"-"(x, y);

        assert(z1.value.flattened == [11.0, 22.0, 13.0, 24.0, 15.0, 26.0, 17.0, 28.0]);
        assert(z2.value.flattened == [-9.0, -18.0, -7.0, -16.0, -5.0, -14.0, -3.0, -12.0]);
    }

    unittest
    {
        auto x = tensor!([0, 2])([1.0, 2.0, 3.0, 4.0]);
        auto y = tensor!([2])([10.0, 20.0]);
        auto z = broadcastOp!"+"(x, y);
        assert(z.shape == [2, 2]);
        assert(z.value == [[11.0, 22.0], [13.0, 24.0]]);

        z.backward();

        import std : text;

        assert(x.grads == [[1.0, 1.0], [1.0, 1.0]], text("x.grads: ", x.grads, " != [[1.0, 1.0], [1.0, 1.0]]"));
        assert(y.grads == [2.0, 2.0], text("y.grads: ", y.grads, " != [2.0, 2.0]"));
    }

    unittest
    {
        auto x = tensor!([0, 2])([1.0, 2.0, 3.0, 4.0]);
        auto y = tensor!([2])([10.0, 20.0]);
        auto z = broadcastOp!"-"(x, y);
        assert(z.shape == [2, 2]);
        assert(z.value == [[-9.0, -18.0], [-7.0, -16.0]]);

        z.backward();

        import std : text;

        assert(x.grads == [[1.0, 1.0], [1.0, 1.0]], text("x.grads: ", x.grads, " != [[1.0, 1.0], [1.0, 1.0]]"));
        assert(y.grads == [-2.0, -2.0], text("y.grads: ", y.grads, " != [-2.0, -2.0]"));
    }

    unittest
    {
        auto x = tensor!([0, 2])([1.0, 2.0, 3.0, 4.0]);
        auto y = (1.0 / x.shape[0]) * batchSum(x); // mean
        auto z = broadcastOp!"-"(x, y);
        assert(z.shape == [2, 2]);
        assert(z.value == [[-1.0, -1.0], [1.0, 1.0]]);

        z.backward();

        assert(x.grads == [[0.0, 0.0], [0.0, 0.0]]);
    }

    unittest
    {
        auto x = tensor!([0, 2, 2])([1.0, 2.0, 3.0, 4.0]);
        auto y = tensor!([2], UseGradient.no)([10.0, 20.0]);

        auto z = broadcastOp!"+"(x, y);
        auto w = broadcastOp!"-"(x, y);
        z.backward();
        w.backward();
    }

    unittest
    {
        auto x = tensor!([0, 2, 2], UseGradient.no)([1.0, 2.0, 3.0, 4.0]);
        auto y = tensor!([2])([10.0, 20.0]);

        auto z = broadcastOp!"+"(x, y);
        auto w = broadcastOp!"-"(x, y);
        z.backward();
        w.backward();
    }

    unittest
    {
        auto x = tensor!([0, 2, 2], UseGradient.no)([1.0, 2.0, 3.0, 4.0]);
        auto y = tensor!([2], UseGradient.no)([10.0, 20.0]);

        auto z = broadcastOp!"+"(x, y);
        auto w = broadcastOp!"-"(x, y);

        static assert(!canBackward!(typeof(z)));
        static assert(!canBackward!(typeof(w)));
    }

    template broadcastOp(string op)
    if (op == "*")
    {
        /+
        [N, C, W, H] * [C, W, H]
        +/
        auto broadcastOp(T, size_t[] Shape1, UseGradient useGrad1, size_t[] Shape2, UseGradient useGrad2)(
                Tensor!(T, Shape1, useGrad1) x, Tensor!(T, Shape2, useGrad2) y)
            if (Shape1[$ - Shape2.length .. $] == Shape2)
        {
            enum Dim2 = Shape2.length;

            alias binOp = (a, b) => a * b;
            
            auto yv = y.value;
            auto z = x.value.pack!Dim2.map!(a => binOp(a, yv)).fuse();

            static if (useGrad1 || useGrad2)
            {
                static if (useGrad1) x.usedCount++;
                static if (useGrad2) y.usedCount++;
                return new Tensor!(T, Shape1, UseGradient.yes)(z, (grads) {
                    static if (useGrad1)
                    {
                        x.backward((ref xGrads) {
                            foreach (ref t; zip(xGrads.pack!Dim2.flattened, grads.pack!Dim2.flattened))
                            {
                                t[0][] += t[1][] * yv[];
                            }
                        });
                    }
                    static if (useGrad2)
                    {
                        y.backward((ref yGrads) {
                            foreach (ref t; zip(grads.pack!Dim2.flattened, x.value.pack!Dim2.flattened))
                            {
                                yGrads[] += t[0] * t[1];
                            }
                        });
                    }
                });
            }
            else
            {
                return new Tensor!(T, Shape1, UseGradient.no)(z);
            }
        }
    }

    unittest
    {
        auto x = tensor!([0, 2, 2])([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        auto y = tensor!([2, 2])([0.2, 0.4, 0.6, 0.8]);

        auto z = broadcastOp!"*"(x, y);

        import std.math : isClose;

        assert(z.value[0, 0, 0].isClose(1.0 * 0.2));
        assert(z.value[0, 0, 1].isClose(2.0 * 0.4));
        assert(z.value[0, 1, 0].isClose(3.0 * 0.6));
        assert(z.value[0, 1, 1].isClose(4.0 * 0.8));
        assert(z.value[1, 0, 0].isClose(5.0 * 0.2));
        assert(z.value[1, 0, 1].isClose(6.0 * 0.4));
        assert(z.value[1, 1, 0].isClose(7.0 * 0.6));
        assert(z.value[1, 1, 1].isClose(8.0 * 0.8));

        z.backward();

        assert(x.grads[0, 0, 0].isClose(0.2));
        assert(x.grads[0, 0, 1].isClose(0.4));
        assert(x.grads[0, 1, 0].isClose(0.6));
        assert(x.grads[0, 1, 1].isClose(0.8));
        assert(x.grads[1, 0, 0].isClose(0.2));
        assert(x.grads[1, 0, 1].isClose(0.4));
        assert(x.grads[1, 1, 0].isClose(0.6));
        assert(x.grads[1, 1, 1].isClose(0.8));

        assert(y.grads[0, 0] == 1.0 + 5.0);
        assert(y.grads[0, 1] == 2.0 + 6.0);
        assert(y.grads[1, 0] == 3.0 + 7.0);
        assert(y.grads[1, 1] == 4.0 + 8.0);
    }

    unittest
    {
        auto x = tensor!([0, 2, 2])([1.0, 2.0, 3.0, 4.0]);
        auto y = tensor!([2], UseGradient.no)([10.0, 20.0]);

        auto z = broadcastOp!"*"(x, y);
        z.backward();
    }

    unittest
    {
        auto x = tensor!([0, 2, 2], UseGradient.no)([1.0, 2.0, 3.0, 4.0]);
        auto y = tensor!([2])([10.0, 20.0]);

        auto z = broadcastOp!"*"(x, y);
        z.backward();
    }

    unittest
    {
        auto x = tensor!([0, 2, 2], UseGradient.no)([1.0, 2.0, 3.0, 4.0]);
        auto y = tensor!([2], UseGradient.no)([10.0, 20.0]);

        auto z = broadcastOp!"*"(x, y);

        static assert(!canBackward!(typeof(z)));
    }
}

version (all) // multicastOp
{
    template multicastOp(string op)
    if (op == "+" || op == "-")
    {
        /+
        [N, C, W, H] + [N, C]
        [N, C, W, H] - [N, C]
        +/
        auto multicastOp(T, size_t[] Shape1, UseGradient useGrad1, size_t[] Shape2, UseGradient useGrad2)(
                Tensor!(T, Shape1, useGrad1) x, Tensor!(T, Shape2, useGrad2) y)
            if (Shape1[0 .. trimRightOneDims(Shape2).length] == trimRightOneDims(Shape2))
        {
            enum Dim2 = trimRightOneDims(Shape2).length;

            auto yv = y.value;

            auto z = slice(x.value);
            foreach (t; zip(z.ipack!Dim2.flattened, yv.flattened))
            {
                static if (op == "+")
                    t[0][] += t[1];
                else static if (op == "-")
                    t[0][] -= t[1];
            }

            static if (useGrad1 || useGrad2)
            {
                static if (useGrad1) x.usedCount++;
                static if (useGrad2) y.usedCount++;
                return new Tensor!(T, Shape1, UseGradient.yes)(z, (grads) {
                    static if (useGrad1)
                    {
                        x.backward(grads);
                    }
                    static if (useGrad2)
                    {
                        y.backward((ref yGrads) {
                            import mir.math.sum : mirsum = sum;

                            static if (op == "+")
                                yGrads[] += grads.ipack!Dim2.map!(a => mirsum(a));
                            else
                                yGrads[] -= grads.ipack!Dim2.map!(a => mirsum(a));
                        });
                    }
                });
            }
            else
            {
                return new Tensor!(T, Shape1, UseGradient.no)(z);
            }
        }
    }

    unittest
    {
        auto x = tensor!([0, 2, 2], UseGradient.no)([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        auto y = tensor!([0], UseGradient.no)([10.0, 20.0]);

        auto z1 = multicastOp!"+"(x, y);
        auto z2 = multicastOp!"-"(x, y);

        assert(z1.value.flattened == [11.0, 12.0, 13.0, 14.0, 25.0, 26.0, 27.0, 28.0]);
        assert(z2.value.flattened == [-9.0, -8.0, -7.0, -6.0, -15.0, -14.0, -13.0, -12.0]);
        
        static assert(!canBackward!(typeof(z1)));
        static assert(!canBackward!(typeof(z2)));
    }

    unittest
    {
        auto x = tensor!([0, 2])([1.0, 2.0, 3.0, 4.0]);
        auto y = tensor!([0])([10.0, 20.0]);
        auto z = multicastOp!"+"(x, y);
        assert(z.shape == [2, 2]);
        assert(z.value == [[11.0, 12.0], [23.0, 24.0]]);

        z.backward();

        import std : text;

        assert(x.grads == [[1.0, 1.0], [1.0, 1.0]], text("x.grads: ", x.grads, " != [[1.0, 1.0], [1.0, 1.0]]"));
        assert(y.grads == [2.0, 2.0], text("y.grads: ", y.grads, " != [2.0, 2.0]"));
    }

    unittest
    {
        auto x = tensor!([0, 2])([1.0, 2.0, 3.0, 4.0]);
        auto y = tensor!([0])([10.0, 20.0]);
        auto z = multicastOp!"-"(x, y);
        assert(z.shape == [2, 2]);
        assert(z.value == [[-9.0, -8.0], [-17.0, -16.0]]);

        z.backward();

        import std : text;

        assert(x.grads == [[1.0, 1.0], [1.0, 1.0]], text("x.grads: ", x.grads, " != [[1.0, 1.0], [1.0, 1.0]]"));
        assert(y.grads == [-2.0, -2.0], text("y.grads: ", y.grads, " != [-2.0, -2.0]"));
    }

    unittest
    {
        auto x = tensor!([0, 2, 2])([1.0, 2.0, 3.0, 4.0]);
        auto y = tensor!([0, 2], UseGradient.no)([10.0, 20.0]);

        auto z = multicastOp!"+"(x, y);
        auto w = multicastOp!"-"(x, y);
        z.backward();
        w.backward();
    }

    unittest
    {
        auto x = tensor!([0, 2, 2], UseGradient.no)([1.0, 2.0, 3.0, 4.0]);
        auto y = tensor!([0, 2])([10.0, 20.0]);

        auto z = multicastOp!"+"(x, y);
        auto w = multicastOp!"-"(x, y);
        z.backward();
        w.backward();
    }

    unittest
    {
        // remove the average for each batch
        auto x = tensor!([0, 2, 2])([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        auto y = mean(x);

        auto z = multicastOp!"-"(x, y);

        assert(z.value.flattened == [-1.5, -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5]);
    }


    template multicastOp(string op)
    if (op == "*")
    {
        /+
        [N, C, W, H] * [N, C]
        +/
        auto multicastOp(T, size_t[] Shape1, UseGradient useGrad1, size_t[] Shape2, UseGradient useGrad2)(
                Tensor!(T, Shape1, useGrad1) x, Tensor!(T, Shape2, useGrad2) y)
            if (Shape1[0 .. trimRightOneDims(Shape2).length] == trimRightOneDims(Shape2))
        {
            enum Dim2 = trimRightOneDims(Shape2).length;

            auto yv = y.value;

            auto z = slice(x.value);
            foreach (t; zip(z.ipack!Dim2.flattened, yv.flattened))
            {
                t[0][] *= t[1];
            }

            static if (useGrad1 || useGrad2)
            {
                static if (useGrad1) x.usedCount++;
                static if (useGrad2) y.usedCount++;
                return new Tensor!(T, Shape1, UseGradient.yes)(z, (grads) {
                    static if (useGrad1)
                    {
                        x.backward((ref xGrads) {
                            foreach (t; zip(xGrads.ipack!Dim2.flattened, grads.ipack!Dim2.flattened, y.value.flattened))
                            {
                                t[0][] += t[1][] * t[2];
                            }
                        });
                    }
                    static if (useGrad2)
                    {
                        y.backward((ref yGrads) {
                            import mir.math.sum : mirsum = sum;

                            yGrads[] += (grads * x.value).ipack!Dim2.map!(a => mirsum(a));
                        });
                    }
                });
            }
            else
            {
                return new Tensor!(T, Shape1, UseGradient.no)(z);
            }
        }
    }

    unittest
    {
        auto x = tensor!([0, 2, 2])([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        auto y = tensor!([0])([2.0, 3.0]);

        auto z = multicastOp!"*"(x, y);

        import std.math : isClose;

        assert(z.value[0, 0, 0] == 1.0 * 2);
        assert(z.value[0, 0, 1] == 2.0 * 2);
        assert(z.value[0, 1, 0] == 3.0 * 2);
        assert(z.value[0, 1, 1] == 4.0 * 2);
        assert(z.value[1, 0, 0] == 5.0 * 3);
        assert(z.value[1, 0, 1] == 6.0 * 3);
        assert(z.value[1, 1, 0] == 7.0 * 3);
        assert(z.value[1, 1, 1] == 8.0 * 3);

        z.backward();

        assert(x.grads[0, 0, 0] == 2.0);
        assert(x.grads[0, 0, 1] == 2.0);
        assert(x.grads[0, 1, 0] == 2.0);
        assert(x.grads[0, 1, 1] == 2.0);
        assert(x.grads[1, 0, 0] == 3.0);
        assert(x.grads[1, 0, 1] == 3.0);
        assert(x.grads[1, 1, 0] == 3.0);
        assert(x.grads[1, 1, 1] == 3.0);

        assert(y.grads[0] == 1.0 + 2.0 + 3.0 + 4.0);
        assert(y.grads[1] == 5.0 + 6.0 + 7.0 + 8.0);
    }

    unittest
    {
        auto x = tensor!([0, 2])([1.0, 2.0, 3.0, 4.0]);
        auto y = tensor!([0], UseGradient.no)([10.0, 20.0]);

        auto z = multicastOp!"*"(x, y);
        z.backward();
    }

    unittest
    {
        auto x = tensor!([0, 2], UseGradient.no)([1.0, 2.0, 3.0, 4.0]);
        auto y = tensor!([0])([10.0, 20.0]);

        auto z = multicastOp!"*"(x, y);
        z.backward();
    }

    unittest
    {
        auto x = tensor!([0, 2], UseGradient.no)([1.0, 2.0, 3.0, 4.0]);
        auto y = tensor!([0], UseGradient.no)([10.0, 20.0]);

        auto z = multicastOp!"*"(x, y);

        static assert(!canBackward!(typeof(z)));
    }
}

version (all) // splitEvenOdd2D
{
    ///
    auto splitEvenOdd2D(size_t axis = 2, T, size_t[] Shape, UseGradient useGrad)(
            Tensor!(T, Shape, useGrad) images)
            if (Shape.length == 4 && (axis == 2 || axis == 3))
    {
        static if (axis == 2)
        {
            static assert(Shape[2] % 2 == 0);
            enum height = Shape[2] / 2;
            enum width = Shape[3];

            auto y1 = images.value[0 .. $, 0 .. $, 0 .. $ - 1, 0 .. $].strided!2(2).slice();
            auto y2 = images.value[0 .. $, 0 .. $, 1 .. $, 0 .. $].strided!2(2).slice();
        }
        else
        {
            static assert(Shape[3] % 2 == 0);
            enum height = Shape[2];
            enum width = Shape[3] / 2;

            auto y1 = images.value[0 .. $, 0 .. $, 0 .. $, 0 .. $ - 1].strided!3(2).slice();
            auto y2 = images.value[0 .. $, 0 .. $, 0 .. $, 1 .. $].strided!3(2).slice();
        }

        enum size_t[] ReturnShape = [Shape[0], Shape[1], height, width];
        static if (useGrad)
        {
            images.usedCount += 2;

            static if (axis == 2)
            {
                return tuple(new Tensor!(T, ReturnShape)(y1, (grads) {
                        images.backward((ref imagesGrads) {
                            imagesGrads[0 .. $, 0 .. $, 0 .. $ - 1, 0 .. $].strided!2(2)[] += grads[];
                        });
                    }), new Tensor!(T, ReturnShape)(y2, (grads) {
                        images.backward((ref imagesGrads) {
                            imagesGrads[0 .. $, 0 .. $, 1 .. $, 0 .. $].strided!2(2)[] += grads[];
                        });
                    }));
            }
            else
            {
                return tuple(new Tensor!(T, ReturnShape)(y1, (grads) {
                        images.backward((ref imagesGrads) {
                            imagesGrads[0 .. $, 0 .. $, 0 .. $, 0 .. $ - 1].strided!3(2)[] += grads[];
                        });
                    }), new Tensor!(T, ReturnShape)(y2, (grads) {
                        images.backward((ref imagesGrads) {
                            imagesGrads[0 .. $, 0 .. $, 0 .. $, 1 .. $].strided!3(2)[] += grads[];
                        });
                    }));
            }
        }
        else
        {
            // dfmt off
            return tuple(
                new Tensor!(T, ReturnShape, UseGradient.no)(y1),
                new Tensor!(T, ReturnShape, UseGradient.no)(y2)
                );
            // dfmt on
        }
    }

    /// ditto
    unittest
    {
        auto x = tensor!([0, 1, 2, 2])([1.0, 2.0, 3.0, 4.0]);

        auto sh = splitEvenOdd2D(x); // split by height
        assert(sh[0].shape == [1, 1, 1, 2]);
        assert(sh[0].value == [[[[1.0, 2.0]]]]);
        assert(sh[1].shape == [1, 1, 1, 2]);
        assert(sh[1].value == [[[[3.0, 4.0]]]]);

        sh[0].backward();
        sh[1].backward();

        auto sw = splitEvenOdd2D!3(x); // split by width
        assert(sw[0].shape == [1, 1, 2, 1]);
        assert(sw[0].value == [[[[1.0], [3.0]]]]);
        assert(sw[1].shape == [1, 1, 2, 1]);
        assert(sw[1].value == [[[[2.0], [4.0]]]]);

        sw[0].backward();
        sw[1].backward();
    }

    /// ditto
    unittest
    {
        auto x = tensor!([0, 2, 2, 2], UseGradient.no)([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        auto sh = splitEvenOdd2D!2(x); // split by height
        assert(sh[0].shape == [1, 2, 1, 2]);
        assert(sh[0].value == [[[[1.0, 2.0]], [[5.0, 6.0]]]]);
        assert(sh[1].shape == [1, 2, 1, 2]);
        assert(sh[1].value == [[[[3.0, 4.0]], [[7.0, 8.0]]]]);

        static assert(!canBackward!(typeof(sh)));

        auto sw = splitEvenOdd2D!3(x); // split by width
        assert(sw[0].shape == [1, 2, 2, 1]);
        assert(sw[0].value == [[[[1.0], [3.0]], [[5.0], [7.0]]]]);
        assert(sw[1].shape == [1, 2, 2, 1]);
        assert(sw[1].value == [[[[2.0], [4.0]], [[6.0], [8.0]]]]);

        static assert(!canBackward!(typeof(sw)));
    }
}

version (all) // mergeEvenOdd2D
{
    auto mergeEvenOdd2D(size_t axis = 2, T, size_t[] Shape, UseGradient useGrad1, UseGradient useGrad2)(
            Tensor!(T, Shape, useGrad1) even, Tensor!(T, Shape, useGrad2) odd)
            if (Shape.length == 4)
    {
        static if (axis == 2)
        {
            enum height = Shape[2] * 2;
            enum width = Shape[3];
        }
        else
        {
            enum height = Shape[2];
            enum width = Shape[3] * 2;
        }
        enum size_t[] ReturnShape = [Shape[0], Shape[1], height, width];

        static if (Shape[0] == 0)
            const batchSize = even.shape[0];
        else
            enum batchSize = Shape[0];

        auto y = slice!T(batchSize, Shape[1], height, width);
        static if (axis == 2)
        {
            y[0 .. $, 0 .. $, 0 .. $ - 1, 0 .. $].strided!2(2)[] = even.value[];
            y[0 .. $, 0 .. $, 1 .. $, 0 .. $].strided!2(2)[] = odd.value[];
        }
        else
        {
            y[0 .. $, 0 .. $, 0 .. $, 0 .. $ - 1].strided!3(2)[] = even.value[];
            y[0 .. $, 0 .. $, 0 .. $, 1 .. $].strided!3(2)[] = odd.value[];
        }

        static if (useGrad1 || useGrad2)
        {
            static if (useGrad1)
                even.usedCount++;
            static if (useGrad2)
                odd.usedCount++;

            return new Tensor!(T, ReturnShape)(y, (grads) {
                static if (useGrad1)
                {
                    even.backward((ref evenGrads) {
                        static if (axis == 2)
                            evenGrads[] = grads[0 .. $, 0 .. $, 0 .. $ - 1, 0 .. $].strided!2(2);
                        else
                            evenGrads[] = grads[0 .. $, 0 .. $, 0 .. $, 0 .. $ - 1].strided!3(2);
                    });
                }
                static if (useGrad2)
                {
                    odd.backward((ref oddGrads) {
                        static if (axis == 2)
                            oddGrads[] = grads[0 .. $, 0 .. $, 1 .. $, 0 .. $].strided!2(2);
                        else
                            oddGrads[] = grads[0 .. $, 0 .. $, 0 .. $, 1 .. $].strided!3(2);
                    });
                }
            });
        }
        else
        {
            return new Tensor!(T, ReturnShape, UseGradient.no)(y);
        }
    }

    unittest
    {
        auto x = tensor!([0, 1, 2, 2])([1.0, 2.0, 3.0, 4.0]);
        auto s = splitEvenOdd2D(x);
        auto m = mergeEvenOdd2D(s.expand);

        assert(x.value == m.value);

        m.backward();
    }

    unittest
    {
        auto x = tensor!([0, 1, 2, 2])([1.0, 2.0, 3.0, 4.0]);
        auto s = splitEvenOdd2D!3(x);
        auto m = mergeEvenOdd2D!3(s.expand);

        assert(x.value == m.value);

        m.backward();
    }

    unittest
    {
        auto x = tensor!([0, 1, 2, 2], UseGradient.no)([1.0, 2.0, 3.0, 4.0]);
        auto s = splitEvenOdd2D!2(x);
        auto m = mergeEvenOdd2D!2(s.expand);

        assert(x.value == m.value);

        static assert(!canBackward!(typeof(m)));
    }

    unittest
    {
        auto x = tensor!([0, 1, 2, 2], UseGradient.no)([1.0, 2.0, 3.0, 4.0]);
        auto s = splitEvenOdd2D!3(x);
        auto m = mergeEvenOdd2D!3(s.expand);

        assert(x.value == m.value);

        static assert(!canBackward!(typeof(m)));
    }

    unittest
    {
        auto x = tensor!([0, 1, 2, 2])([1.0, 2.0, 3.0, 4.0]);
        auto y = tensor!([0, 1, 2, 2], UseGradient.no)([1.0, 2.0, 3.0, 4.0]);
        auto m = mergeEvenOdd2D(x, y);

        assert(m.shape == [1, 1, 4, 2]);
        assert(m.value[0, 0, 0, 0] == 1.0);
        assert(m.value[0, 0, 0, 1] == 2.0);
        assert(m.value[0, 0, 1, 0] == 1.0);
        assert(m.value[0, 0, 1, 1] == 2.0);
        assert(m.value[0, 0, 2, 0] == 3.0);
        assert(m.value[0, 0, 2, 1] == 4.0);
        assert(m.value[0, 0, 3, 0] == 3.0);
        assert(m.value[0, 0, 3, 1] == 4.0);

        m.backward();
    }

    unittest
    {
        auto x = tensor!([0, 1, 2, 2], UseGradient.no)([1.0, 2.0, 3.0, 4.0]);
        auto y = tensor!([0, 1, 2, 2], UseGradient.no)([1.0, 2.0, 3.0, 4.0]);
        auto m = mergeEvenOdd2D!3(x, y);

        assert(m.shape == [1, 1, 2, 4]);
        assert(m.value[0, 0, 0, 0] == 1.0);
        assert(m.value[0, 0, 0, 1] == 1.0);
        assert(m.value[0, 0, 0, 2] == 2.0);
        assert(m.value[0, 0, 0, 3] == 2.0);
        assert(m.value[0, 0, 1, 0] == 3.0);
        assert(m.value[0, 0, 1, 1] == 3.0);
        assert(m.value[0, 0, 1, 2] == 4.0);
        assert(m.value[0, 0, 1, 3] == 4.0);
    }
}

version (all) // concat2D
{
    auto concat2D(size_t axis = 1, T, U)(T x, U y)
    if (isTensor!T && isTensor!U)
    {
        static assert(axis == 1, "not implement");

        enum S1 = T.staticShape;
        enum S2 = U.staticShape;
        static assert(S1.length == 4);
        static assert(S2.length == 4);
        static assert(S1[2 .. 4] == S2[2 .. 4]);
        assert(x.shape[0] == y.shape[0]);

        static if (is(T : Tensor!(E, T.staticShape), E))
        {
            alias ElementType = E;
        }
        else static if (is(T : Tensor!(E, T.staticShape, UseGradient.no), E))
        {
            alias ElementType = E;
        }
        else
        {
            static assert(false);
        }

        enum size_t[4] ReturnShape = [S1[0], S1[1] + S2[1], S1[2], S1[3]];
        
        auto z = uninitSlice!ElementType(x.shape[0], S1[1] + S2[1], S1[2], S1[3]);
        z[0 .. $, 0 .. S1[1], 0 .. $, 0 .. $] = x.value;
        z[0 .. $, S1[1] .. $, 0 .. $, 0 .. $] = y.value;

        static if (canBackward!T || canBackward!U)
        {
            static if (canBackward!T)
                x.usedCount++;
            static if (canBackward!U)
                y.usedCount++;

            return new Tensor!(E, ReturnShape)(z, (grads) {
                static if (canBackward!T)
                {
                    x.backward((ref xGrads) {
                        xGrads[] += grads[0 .. $, 0 .. S1[1], 0 .. $, 0 .. $];
                    });
                }
                static if (canBackward!U)
                {
                    y.backward((ref yGrads) {
                        yGrads[] += grads[0 .. $, S1[1] .. $, 0 .. $, 0 .. $];
                    });
                }
            });
        }
        else
        {
            return new Tensor!(E, ReturnShape, UseGradient.no)(z);
        }
    }

    unittest
    {
        auto x = tensor!([0, 1, 2, 2])([1.0, 2.0, 3.0, 4.0]);
        auto y = tensor!([0, 2, 2, 2])([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        auto z = concat2D(x, y);

        assert(z.shape == [1, 3, 2, 2]);
        assert(z.value[0, 0, 0, 0] == 1.0);
        assert(z.value[0, 0, 0, 1] == 2.0);
        assert(z.value[0, 0, 1, 0] == 3.0);
        assert(z.value[0, 0, 1, 1] == 4.0);
        assert(z.value[0, 1, 0, 0] == 1.0);
        assert(z.value[0, 1, 0, 1] == 2.0);
        assert(z.value[0, 1, 1, 0] == 3.0);
        assert(z.value[0, 1, 1, 1] == 4.0);
        assert(z.value[0, 2, 0, 0] == 5.0);
        assert(z.value[0, 2, 0, 1] == 6.0);
        assert(z.value[0, 2, 1, 0] == 7.0);
        assert(z.value[0, 2, 1, 1] == 8.0);

        z.backward();
    }
    
    unittest
    {
        auto x = tensor!([0, 1, 3, 1], UseGradient.no)([1.0, 2.0, 3.0]);
        auto y = tensor!([0, 1, 3, 1])([1.0, 2.0, 3.0]);
        auto z = concat2D(x, y);

        assert(z.shape == [1, 2, 3, 1]);
        assert(z.value[0, 0, 0, 0] == 1.0);
        assert(z.value[0, 0, 1, 0] == 2.0);
        assert(z.value[0, 0, 2, 0] == 3.0);
        assert(z.value[0, 1, 0, 0] == 1.0);
        assert(z.value[0, 1, 1, 0] == 2.0);
        assert(z.value[0, 1, 2, 0] == 3.0);

        z.backward();
    }
    
    unittest
    {
        auto x = tensor!([0, 1, 3, 1])([1.0, 2.0, 3.0]);
        auto y = tensor!([0, 1, 3, 1], UseGradient.no)([1.0, 2.0, 3.0]);
        auto z = concat2D(x, y);

        assert(z.shape == [1, 2, 3, 1]);
        assert(z.value[0, 0, 0, 0] == 1.0);
        assert(z.value[0, 0, 1, 0] == 2.0);
        assert(z.value[0, 0, 2, 0] == 3.0);
        assert(z.value[0, 1, 0, 0] == 1.0);
        assert(z.value[0, 1, 1, 0] == 2.0);
        assert(z.value[0, 1, 2, 0] == 3.0);

        z.backward();
    }
    
    unittest
    {
        auto x = tensor!([0, 1, 3, 1], UseGradient.no)([1.0, 2.0, 3.0]);
        auto y = tensor!([0, 1, 3, 1], UseGradient.no)([1.0, 2.0, 3.0]);
        auto z = concat2D(x, y);

        assert(z.shape == [1, 2, 3, 1]);
        assert(z.value[0, 0, 0, 0] == 1.0);
        assert(z.value[0, 0, 1, 0] == 2.0);
        assert(z.value[0, 0, 2, 0] == 3.0);
        assert(z.value[0, 1, 0, 0] == 1.0);
        assert(z.value[0, 1, 1, 0] == 2.0);
        assert(z.value[0, 1, 2, 0] == 3.0);

        static assert(!canBackward!(typeof(z)));
    }
}

version (all) // projection1D
{
    auto projection1D(size_t axis, T, size_t[] ShapeW, UseGradient useGradW, size_t[] ShapeX, UseGradient useGradX)(Tensor!(T, ShapeX, useGradX) x, Tensor!(T, ShapeW, useGradW) w)
    if (ShapeX.length == 4 && (axis == 2 || axis == 3) && ShapeW.length == 2 && ShapeX[axis] == ShapeW[0])
    {
        enum H = axis == 2 ? ShapeW[1] : ShapeX[2];
        enum W = axis == 3 ? ShapeW[1] : ShapeX[3];
        auto y = uninitSlice!T(x.shape[0], x.shape[1], H, W);

        import mir.blas : gemm;

        auto tx = x.value.ipack!2.flattened;
        auto ty = y.ipack!2.flattened;
        static if (axis == 2)
        {
            auto tw = w.value.transposed;
            foreach (t; zip(tx, ty))
            {
                gemm(T(1), tw, t[0], T(0), t[1]);
            }
        }
        else static if (axis == 3)
        {
            foreach (t; zip(tx, ty))
            {
                gemm(T(1), t[0], w.value, T(0), t[1]);
            }
        }

        enum size_t[4] ReturnShape = [ShapeX[0], ShapeX[1], H, W];
        static if (useGradW || useGradX)
        {
            static if (useGradW)
                w.usedCount++;
            static if (useGradX)
                x.usedCount++;

            return new Tensor!(T, ReturnShape)(y, (grads) {
                static if (useGradW)
                {
                    w.backward((ref wGrads) {
                        auto tx = x.value.ipack!2.flattened;
                        auto tg = grads.ipack!2.flattened;
                        foreach (t; zip(tx, tg))
                        {
                            static if (axis == 2)
                            {
                                gemm(T(1), t[0], t[1].transposed, T(1), wGrads);
                            }
                            else static if (axis == 3)
                            {
                                gemm(T(1), t[0].transposed, t[1], T(1), wGrads);
                            }
                        }
                    });
                }
                static if (useGradX)
                {
                    x.backward((ref xGrads) {
                        auto txg = xGrads.ipack!2.flattened;
                        auto tg = grads.ipack!2.flattened;
                        foreach (t; zip(txg, tg))
                        {
                            static if (axis == 2)
                            {
                                gemm(T(1), w.value, t[1], T(1), t[0]);
                            }
                            else static if (axis == 3)
                            {
                                gemm(T(1), w.value, t[1].transposed, T(1), t[0].transposed);
                            }
                        }
                    });
                }
            });
        }
        else
        {
            return new Tensor!(T, ReturnShape, UseGradient.no)(y);
        }
    }
    
    auto projection1D(size_t axis, T, size_t[] ShapeW, UseGradient useGradW, size_t[] ShapeX, UseGradient useGradX)(Tensor!(T, ShapeX, useGradX) x, Tensor!(T, ShapeW, useGradW) w)
    if (ShapeX.length == 3 && (axis == 1 || axis == 2) && ShapeW.length == 2 && ShapeX[axis] == ShapeW[0])
    {
        enum H = axis == 1 ? ShapeW[1] : ShapeX[1];
        enum W = axis == 2 ? ShapeW[1] : ShapeX[2];
        auto y = uninitSlice!T(x.shape[0], H, W);

        import mir.blas : gemm;

        auto tx = x.value.ipack!1.flattened;
        auto ty = y.ipack!1.flattened;
        static if (axis == 1)
        {
            auto tw = w.value.transposed;
            foreach (t; zip(tx, ty))
            {
                gemm(T(1), tw, t[0], T(0), t[1]);
            }
        }
        else static if (axis == 2)
        {
            foreach (t; zip(tx, ty))
            {
                gemm(T(1), t[0], w.value, T(0), t[1]);
            }
        }

        enum size_t[3] ReturnShape = [ShapeX[0], H, W];
        static if (useGradW || useGradX)
        {
            static if (useGradW)
                w.usedCount++;
            static if (useGradX)
                x.usedCount++;

            return new Tensor!(T, ReturnShape)(y, (grads) {
                static if (useGradW)
                {
                    w.backward((ref wGrads) {
                        auto tx = x.value.ipack!1.flattened;
                        auto tg = grads.ipack!1.flattened;
                        foreach (t; zip(tx, tg))
                        {
                            static if (axis == 1)
                            {
                                gemm(T(1), t[0], t[1].transposed, T(1), wGrads);
                            }
                            else static if (axis == 2)
                            {
                                gemm(T(1), t[0].transposed, t[1], T(1), wGrads);
                            }
                        }
                    });
                }
                static if (useGradX)
                {
                    x.backward((ref xGrads) {
                        auto txg = xGrads.ipack!1.flattened;
                        auto tg = grads.ipack!1.flattened;
                        foreach (t; zip(txg, tg))
                        {
                            static if (axis == 1)
                            {
                                gemm(T(1), w.value, t[1], T(1), t[0]);
                            }
                            else static if (axis == 2)
                            {
                                gemm(T(1), w.value, t[1].transposed, T(1), t[0].transposed);
                            }
                        }
                    });
                }
            });
        }
        else
        {
            return new Tensor!(T, ReturnShape, UseGradient.no)(y);
        }
    }

    // Shape.length == 4
    unittest
    {
        auto x = tensor!([1, 1, 2, 2])([1.0f, 2.0f, 3.0f, 4.0f]);
        auto w = tensor!([2, 3])([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]);

        auto y = projection1D!2(x, w);
        assert(y.shape == [1, 1, 3, 2]);
        assert(y.value[0, 0, 0, 0] == 13);
        assert(y.value[0, 0, 0, 1] == 18);
        assert(y.value[0, 0, 1, 0] == 17);
        assert(y.value[0, 0, 1, 1] == 24);
        assert(y.value[0, 0, 2, 0] == 21);
        assert(y.value[0, 0, 2, 1] == 30);

        y.backward();

        assert(x.grads[0, 0, 0, 0] == 6);
        assert(x.grads[0, 0, 0, 1] == 6);
        assert(x.grads[0, 0, 1, 0] == 15);
        assert(x.grads[0, 0, 1, 1] == 15);

        assert(w.grads[0, 0] == 3);
        assert(w.grads[0, 1] == 3);
        assert(w.grads[0, 2] == 3);
        assert(w.grads[1, 0] == 7);
        assert(w.grads[1, 1] == 7);
        assert(w.grads[1, 2] == 7);
    }

    unittest
    {
        auto x = tensor!([1, 1, 2, 3])([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]);
        auto w = tensor!([2, 3])([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]);

        auto y = projection1D!2(x, w);
        assert(y.shape == [1, 1, 3, 3]);
        assert(y.value[0, 0, 0, 0] == 17);
        assert(y.value[0, 0, 0, 1] == 22);
        assert(y.value[0, 0, 0, 2] == 27);
        assert(y.value[0, 0, 1, 0] == 22);
        assert(y.value[0, 0, 1, 1] == 29);
        assert(y.value[0, 0, 1, 2] == 36);
        assert(y.value[0, 0, 2, 0] == 27);
        assert(y.value[0, 0, 2, 1] == 36);
        assert(y.value[0, 0, 2, 2] == 45);
    }

    unittest
    {
        auto x = tensor!([1, 1, 2, 2])([1.0f, 2.0f, 3.0f, 4.0f]);
        auto w = tensor!([2, 3])([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]);

        auto y = projection1D!3(x, w);
        assert(y.shape == [1, 1, 2, 3]);
        assert(y.value[0, 0, 0, 0] == 9);
        assert(y.value[0, 0, 0, 1] == 12);
        assert(y.value[0, 0, 0, 2] == 15);
        assert(y.value[0, 0, 1, 0] == 19);
        assert(y.value[0, 0, 1, 1] == 26);
        assert(y.value[0, 0, 1, 2] == 33);

        y.backward();

        assert(x.grads[0, 0, 0, 0] == 6);
        assert(x.grads[0, 0, 0, 1] == 15);
        assert(x.grads[0, 0, 1, 0] == 6);
        assert(x.grads[0, 0, 1, 1] == 15);

        assert(w.grads[0, 0] == 4);
        assert(w.grads[0, 1] == 4);
        assert(w.grads[0, 2] == 4);
        assert(w.grads[1, 0] == 6);
        assert(w.grads[1, 1] == 6);
        assert(w.grads[1, 2] == 6);
    }

    unittest
    {
        auto x = tensor!([1, 1, 3, 2])([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]);
        auto w = tensor!([2, 3])([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]);

        auto y = projection1D!3(x, w);
        assert(y.shape == [1, 1, 3, 3]);
        assert(y.value[0, 0, 0, 0] == 9);
        assert(y.value[0, 0, 0, 1] == 12);
        assert(y.value[0, 0, 0, 2] == 15);
        assert(y.value[0, 0, 1, 0] == 19);
        assert(y.value[0, 0, 1, 1] == 26);
        assert(y.value[0, 0, 1, 2] == 33);
        assert(y.value[0, 0, 2, 0] == 29);
        assert(y.value[0, 0, 2, 1] == 40);
        assert(y.value[0, 0, 2, 2] == 51);
    }

    // Shape.length == 3
    unittest
    {
        auto x = tensor!([1, 2, 2])([1.0f, 2.0f, 3.0f, 4.0f]);
        auto w = tensor!([2, 3])([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]);

        auto y = projection1D!1(x, w);
        assert(y.shape == [1, 3, 2]);
        assert(y.value[0, 0, 0] == 13);
        assert(y.value[0, 0, 1] == 18);
        assert(y.value[0, 1, 0] == 17);
        assert(y.value[0, 1, 1] == 24);
        assert(y.value[0, 2, 0] == 21);
        assert(y.value[0, 2, 1] == 30);

        y.backward();

        assert(x.grads[0, 0, 0] == 6);
        assert(x.grads[0, 0, 1] == 6);
        assert(x.grads[0, 1, 0] == 15);
        assert(x.grads[0, 1, 1] == 15);

        assert(w.grads[0, 0] == 3);
        assert(w.grads[0, 1] == 3);
        assert(w.grads[0, 2] == 3);
        assert(w.grads[1, 0] == 7);
        assert(w.grads[1, 1] == 7);
        assert(w.grads[1, 2] == 7);
    }

    unittest
    {
        auto x = tensor!([1, 2, 3])([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]);
        auto w = tensor!([2, 3])([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]);

        auto y = projection1D!1(x, w);
        assert(y.shape == [1, 3, 3]);
        assert(y.value[0, 0, 0] == 17);
        assert(y.value[0, 0, 1] == 22);
        assert(y.value[0, 0, 2] == 27);
        assert(y.value[0, 1, 0] == 22);
        assert(y.value[0, 1, 1] == 29);
        assert(y.value[0, 1, 2] == 36);
        assert(y.value[0, 2, 0] == 27);
        assert(y.value[0, 2, 1] == 36);
        assert(y.value[0, 2, 2] == 45);
    }

    unittest
    {
        auto x = tensor!([1, 2, 2])([1.0f, 2.0f, 3.0f, 4.0f]);
        auto w = tensor!([2, 3])([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]);

        auto y = projection1D!2(x, w);
        assert(y.shape == [1, 2, 3]);
        assert(y.value[0, 0, 0] == 9);
        assert(y.value[0, 0, 1] == 12);
        assert(y.value[0, 0, 2] == 15);
        assert(y.value[0, 1, 0] == 19);
        assert(y.value[0, 1, 1] == 26);
        assert(y.value[0, 1, 2] == 33);

        y.backward();

        assert(x.grads[0, 0, 0] == 6);
        assert(x.grads[0, 0, 1] == 15);
        assert(x.grads[0, 1, 0] == 6);
        assert(x.grads[0, 1, 1] == 15);

        assert(w.grads[0, 0] == 4);
        assert(w.grads[0, 1] == 4);
        assert(w.grads[0, 2] == 4);
        assert(w.grads[1, 0] == 6);
        assert(w.grads[1, 1] == 6);
        assert(w.grads[1, 2] == 6);
    }

    unittest
    {
        auto x = tensor!([1, 3, 2])([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]);
        auto w = tensor!([2, 3])([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]);

        auto y = projection1D!2(x, w);
        assert(y.shape == [1, 3, 3]);
        assert(y.value[0, 0, 0] == 9);
        assert(y.value[0, 0, 1] == 12);
        assert(y.value[0, 0, 2] == 15);
        assert(y.value[0, 1, 0] == 19);
        assert(y.value[0, 1, 1] == 26);
        assert(y.value[0, 1, 2] == 33);
        assert(y.value[0, 2, 0] == 29);
        assert(y.value[0, 2, 1] == 40);
        assert(y.value[0, 2, 2] == 51);
    }
}

version (all) // conv2D
{
    size_t[] conv2DShape(size_t[] Shape, size_t channel_out, size_t[] kernel_size, size_t[] padding, size_t[] stride, size_t[] dilation)
    {
        import std.math : floor;

        const H_in = Shape[2];
        const W_in = Shape[3];

        const H_out = cast(size_t) floor(real(H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1);
        const W_out = cast(size_t) floor(real(W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1);

        return [Shape[0], channel_out, H_out, W_out];
    }

    unittest
    {
        auto shape = conv2DShape([0, 1, 28, 28], 4, [3, 3], [0, 0], [1, 1], [1, 1]);
        assert(shape == [0, 4, 26, 26]);
    }

    auto conv2D(
        size_t[] padding = [0, 0],
        size_t[] stride = [1, 1],
        size_t[] dilation = [1, 1],
        T,
        size_t[] ShapeX, UseGradient useGradX,
        size_t[] ShapeW, UseGradient useGradW,
        size_t[] ShapeB, UseGradient useGradB
    )(Tensor!(T, ShapeX, useGradX) x, Tensor!(T, ShapeW, useGradW) weights, Tensor!(T, ShapeB, useGradB) bias)
    {
        static assert(padding.length == 2);
        static assert(stride.length == 2);
        static assert(dilation.length == 2);
        static assert(stride == [1, 1], "conv2d : stride is not implemented");
        static assert(dilation == [1, 1], "conv2d : dilation is not implemented");

        static assert(ShapeX.length == 4);
        static assert(ShapeW.length == 4);
        static assert(ShapeB.length == 1);
        static assert(ShapeX[1] == ShapeW[1]);
        static assert(ShapeW[0] == ShapeB[0]);

        enum ReturnShape = conv2DShape(ShapeX, ShapeB[0], ShapeW[2 .. 4], padding, stride, dilation);

        enum C = ShapeX[1];
        enum C_out = ReturnShape[1];
        enum TempH = ShapeX[2] + 2 * padding[0];
        enum TempW = ShapeX[3] + 2 * padding[1];
        enum usePadding = padding[0] != 0 || padding[1] != 0;
        static if (usePadding)
        {
            auto temp = slice!T([C, TempH, TempW], 0);
        }
        auto y = uninitSlice!T(x.shape[0], ReturnShape[1], ReturnShape[2], ReturnShape[3]);

        // prepare im2col
        int err;
        auto ty = y.reshape([x.shape[0], ReturnShape[1], ReturnShape[2] * ReturnShape[3]], err);
        assert(err == 0);
        auto v = uninitSlice!T(ReturnShape[2] * ReturnShape[3], C * ShapeW[2] * ShapeW[3] + 1);
        v[0 .. $, $ - 1 .. $] = 1;
        auto w = uninitSlice!T(C_out, C * ShapeW[2] * ShapeW[3] + 1);
        foreach (i; 0 .. C_out)
        {
            w[i].flattened[0 .. $ - 1] = weights.value[i].flattened;
            w[i].back = bias.value[i];
        }

        foreach (i; 0 .. x.shape[0])
        {
            static if (usePadding)
            {
                temp[0 .. $, padding[0] .. $ - padding[0], padding[1] .. $ - padding[1]] = x.value[i];
                auto wins = temp.windows(C, ShapeW[2], ShapeW[3]);
            }
            else
            {
                auto wins = x.value[i].windows(C, ShapeW[2], ShapeW[3]);
            }
            foreach (t; zip(v.ipack!1, wins.flattened))
            {
                t[0].flattened[0 .. $ - 1] = t[1].flattened;
            }

            import mir.blas : gemm;

            gemm(T(1), v, w.transposed, T(0), ty[i].transposed);
        }
        
        static if (useGradX || useGradW || useGradB)
        {
            static if (useGradX)
                x.usedCount++;
            static if (useGradW)
                weights.usedCount++;
            static if (useGradB)
                bias.usedCount++;

            return new Tensor!(T, ReturnShape)(y, (grads) {
                static if (useGradX)
                {
                    x.backward((ref xGrads) {
                        foreach (i; 0 .. grads.shape[0])
                        {
                            static if (usePadding)
                            {
                                temp.flattened[] = 0;
                                auto wins = temp.windows(ShapeX[1], ShapeW[2], ShapeW[3]);
                            }
                            else
                            {
                                auto wins = xGrads[i].windows(ShapeX[1], ShapeW[2], ShapeW[3]);
                            }
                            foreach (h; 0 .. ReturnShape[2])
                            {
                                foreach (w; 0 .. ReturnShape[3])
                                {
                                    auto tw = wins[0, h, w];
                                    auto tg = grads.transposed!(0, 2, 3, 1)[i, h, w];
                                    foreach (c; 0 .. ReturnShape[1])
                                    {
                                        tw[] += weights.value[c] * tg[c];
                                    }
                                }
                            }
                            static if (usePadding)
                            {
                                xGrads[i][] += temp[0 .. $, padding[0] .. $ - padding[0], padding[1] .. $ - padding[1]];
                            }
                        }
                    });
                }
                static if (useGradW)
                {
                    weights.backward((ref wGrads) {
                        static if (usePadding)
                        {
                            temp.flattened[] = 0;
                        }
                        foreach (i; 0 .. grads.shape[0])
                        {
                            static if (usePadding)
                            {
                                temp[0 .. $, padding[0] .. $ - padding[0], padding[1] .. $ - padding[1]] = x.value[i];
                                auto wins = temp.windows(C, ShapeW[2], ShapeW[3]);
                            }
                            else
                            {
                                auto wins = x.value[i].windows(C, ShapeW[2], ShapeW[3]);
                            }
                            foreach (h; 0 .. ReturnShape[2])
                            {
                                foreach (w; 0 .. ReturnShape[3])
                                {
                                    auto tw = wins[0, h, w];
                                    auto tg = grads.transposed!(0, 2, 3)[i, h, w];
                                    foreach (c; 0 .. ShapeW[0])
                                    {
                                        wGrads[c][] += tg[c] * tw;
                                    }
                                }
                            }
                        }
                    });
                }
                static if (useGradB)
                {
                    bias.backward((ref bGrads) {
                        import mir.math.sum : sum;

                        bGrads[] += grads.transposed!1.ipack!1.map!sum;
                    });
                }
            });
        }
        else
        {
            return new Tensor!(T, ReturnShape, UseGradient.no)(y);
        }
    }

    unittest
    {
        // dfmt off
        auto images = tensor!([0, 1, 5, 5])([
             1.0,  2.0,  3.0,  4.0,  5.0,
             6.0,  7.0,  8.0,  9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0, 25.0]);
        // dfmt on

        // dfmt off
        auto weights = tensor!([2, 1, 3, 3])([
             1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0
            ]);
        // dfmt on

        auto bias = tensor!([2])([1.0, 2.0]);

        auto y = conv2D(images, weights, bias);

        assert(y.shape == [1, 2, 3, 3]);
        assert(y.value[0, 0, 0, 0] == 412);
        assert(y.value[0, 0, 0, 1] == 457);
        assert(y.value[0, 0, 0, 2] == 502);
        assert(y.value[0, 0, 1, 0] == 637);
        assert(y.value[0, 0, 1, 1] == 682);
        assert(y.value[0, 0, 1, 2] == 727);
        assert(y.value[0, 0, 2, 0] == 862);
        assert(y.value[0, 0, 2, 1] == 907);
        assert(y.value[0, 0, 2, 2] == 952);

        assert(y.value[0, 1, 0, 0] == 980);
        assert(y.value[0, 1, 0, 1] == 1106);
        assert(y.value[0, 1, 0, 2] == 1232);
        assert(y.value[0, 1, 1, 0] == 1610);
        assert(y.value[0, 1, 1, 1] == 1736);
        assert(y.value[0, 1, 1, 2] == 1862);
        assert(y.value[0, 1, 2, 0] == 2240);
        assert(y.value[0, 1, 2, 1] == 2366);
        assert(y.value[0, 1, 2, 2] == 2492);

        y.backward();

        assert(images.grads[0, 0, 0, 0] == 11);
        assert(images.grads[0, 0, 0, 1] == 24);
        assert(images.grads[0, 0, 0, 2] == 39);
        assert(images.grads[0, 0, 0, 3] == 28);
        assert(images.grads[0, 0, 0, 4] == 15);
        assert(images.grads[0, 0, 1, 0] == 28);
        assert(images.grads[0, 0, 1, 1] == 60);
        assert(images.grads[0, 0, 1, 2] == 96);
        assert(images.grads[0, 0, 1, 3] == 68);
        assert(images.grads[0, 0, 1, 4] == 36);
        assert(images.grads[0, 0, 2, 0] == 51);
        assert(images.grads[0, 0, 2, 1] == 108);
        assert(images.grads[0, 0, 2, 2] == 171);
        assert(images.grads[0, 0, 2, 3] == 120);
        assert(images.grads[0, 0, 2, 4] == 63);
        assert(images.grads[0, 0, 3, 0] == 40);
        assert(images.grads[0, 0, 3, 1] == 84);
        assert(images.grads[0, 0, 3, 2] == 132);
        assert(images.grads[0, 0, 3, 3] == 92);
        assert(images.grads[0, 0, 3, 4] == 48);
        assert(images.grads[0, 0, 4, 0] == 23);
        assert(images.grads[0, 0, 4, 1] == 48);
        assert(images.grads[0, 0, 4, 2] == 75);
        assert(images.grads[0, 0, 4, 3] == 52);
        assert(images.grads[0, 0, 4, 4] == 27);

        assert(weights.grads[0, 0, 0, 0] == 63);
        assert(weights.grads[0, 0, 0, 1] == 72);
        assert(weights.grads[0, 0, 0, 2] == 81);
        assert(weights.grads[0, 0, 1, 0] == 108);
        assert(weights.grads[0, 0, 1, 1] == 117);
        assert(weights.grads[0, 0, 1, 2] == 126);
        assert(weights.grads[0, 0, 2, 0] == 153);
        assert(weights.grads[0, 0, 2, 1] == 162);
        assert(weights.grads[0, 0, 2, 2] == 171);
        assert(weights.grads[1, 0, 0, 0] == 63);
        assert(weights.grads[1, 0, 0, 1] == 72);
        assert(weights.grads[1, 0, 0, 2] == 81);
        assert(weights.grads[1, 0, 1, 0] == 108);
        assert(weights.grads[1, 0, 1, 1] == 117);
        assert(weights.grads[1, 0, 1, 2] == 126);
        assert(weights.grads[1, 0, 2, 0] == 153);
        assert(weights.grads[1, 0, 2, 1] == 162);
        assert(weights.grads[1, 0, 2, 2] == 171);

        assert(bias.grads[0] == 9);
        assert(bias.grads[1] == 9);
    }
    
    unittest
    {
        // dfmt off
        auto x = tensor!([2, 1, 3, 3])([
            -1.0,  0.0,  1.0,
            0.0,  1.0,  0.0,
            1.0,  0.0, -1.0,
            1.0, -1.0, -0.5,
            -1.0,  1.0, -1.0,
            -0.5, -1.0,  1.0,
        ]);

        auto w = tensor!([1, 1, 3, 3])([
            -0.5,  -0.5,  0.75,
            -0.5,   1.0, -0.5,
            0.75, -0.5, -0.5,
        ]);

        auto b = tensor!([1])([0.0]);
        // dfmt on

        auto y = conv2D!([1, 1])(x, w, b);

        assert(y.shape == [2, 1, 3, 3]);

        assert(y.value[0, 0, 0, 0] == -1.5);
        assert(y.value[0, 0, 0, 1] == -0.5);
        assert(y.value[0, 0, 0, 2] == 1.75);
        assert(y.value[0, 0, 1, 0] == -0.5);
        assert(y.value[0, 0, 1, 1] == 3.5);
        assert(y.value[0, 0, 1, 2] == -0.5);
        assert(y.value[0, 0, 2, 0] == 1.75);
        assert(y.value[0, 0, 2, 1] == -0.5);
        assert(y.value[0, 0, 2, 2] == -1.5);
        
        assert(y.value[1, 0, 0, 0] == 1.5);
        assert(y.value[1, 0, 0, 1] == -2);
        assert(y.value[1, 0, 0, 2] == 1.25);
        assert(y.value[1, 0, 1, 0] == -2);
        assert(y.value[1, 0, 1, 1] == 1.25);
        assert(y.value[1, 0, 1, 2] == -2);
        assert(y.value[1, 0, 2, 0] == 1.25);
        assert(y.value[1, 0, 2, 1] == -2);
        assert(y.value[1, 0, 2, 2] == 1.5);

        y.backward();
    }
}
