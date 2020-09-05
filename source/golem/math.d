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
    Tensor!(T, [ShapeX[0], ShapeW[1]]) linear(T, size_t[2] ShapeX, UseGradient useGradX,
            size_t[2] ShapeW, UseGradient useGradW, size_t[1] ShapeB, UseGradient useGradB)(Tensor!(T, ShapeX, useGradX) x,
            Tensor!(T, ShapeW, useGradW) W, Tensor!(T, ShapeB, useGradB) B)
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

        alias Return = Tensor!(T, [ShapeX[0], OutputDim]);
        alias Value = Return.Value;

        static if (canBackward!(typeof(W)) || canBackward!(typeof(x)) || canBackward!(typeof(B)))
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
}

version (all) // sum
{
    Tensor!(T, [1]) sum(alias mode = "fast", T, size_t[] Shape)(Tensor!(T, Shape) x)
            if (Shape.length == 2 && Shape[1] == 1)
    {
        import mir.math.sum : mirsum = sum;

        auto y = slice!T([1], mirsum!mode(x.value));

        alias Return = typeof(return);
        alias Value = Return.Value;

        x.usedCount++;

        return new Return(y, (Value grad) {
            x.backward((ref xGrads) { xGrads[] += grad; });
        });
    }

    unittest
    {
        auto x = tensor!([4, 1])([1.0f, 2.0f, 3.0f, 4.0f]);
        auto s = sum(x);

        assert(x.grads == [[0.0f], [0.0f], [0.0f], [0.0f]]);
        s.backward();
        assert(x.grads == [[1.0f], [1.0f], [1.0f], [1.0f]]);
    }

    Tensor!(T, [Shape[0], 1]) sum(alias mode = "fast", T, size_t[] Shape)(Tensor!(T, Shape) x)
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
}

version (all) // flatten
{
    Tensor!(T, [Shape[0], elementSize(Shape[1 .. $])]) flatten(T, size_t[] Shape)(Tensor!(T, Shape) x)
    {
        int err;
        auto y = x.value.reshape([x.value.shape[0], -1], err);
        assert(err == 0);

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

    unittest
    {
        auto x = tensor!([2, 2, 2])([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        auto y = flatten(x);
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
}
