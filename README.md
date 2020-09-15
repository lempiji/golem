# golem

WIP

## Features

- Computational graph (autograd)
- A statically size checked slice
- Some friendly error messages
- Simple SGD optimizer

## Examples

```d
import golem;

// statically sized tensor
auto x = tensor!([2, 2])([
        [0.1, 0.2],
        [0.3, 0.4],
    ]);
auto y = tensor!([2, 2])([
        [-0.1, 0.2],
        [0.3, -0.4],
    ]);

auto z = x + y;

assert(z.value[0, 0] == 0.0);
```


### Tensor Shape

```d
// 3 x 2
auto x = tensor!([3, 2])(
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
);
// N x 2
auto y = tensor!([0, 2])([
    [1.0, 2.0],
    [3.0, 4.0],
]);

assert(x.shape == [3, 2]);
assert(y.shape == [2, 2]);

static assert(x.staticShape == [3, 2]);
static assert(y.staticShape == [0, 2]);

assert(x.runtimeShape == [3, 2]);
assert(y.runtimeShape == [2, 2]);

const batchSize = x.shape[0];
```

```d
auto x = tensor!([2, 3])(
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    );
auto y = tensor!([2, 2])([
        [1.0, 2.0],
        [3.0, 4.0],
    ]);
auto z = tensor!([0, 2])([
        [1.0, 2.0],
        [3.0, 4.0],
    ]);

// can not compile
static assert(!__traits(compiles, {
        auto a = x + y;
    }));

// runtime error
assertThrown!AssertError(x + z);
```


### Linear

```d
import golem;

// prepare datasets with dynamic batch sizes
auto data = tensor!([0, 2])([
        [0.1, 0.2],
        [0.1, 0.3],
        [0.15, 0.4],
        [0.2, 0.5],
    ]);
auto label = tensor!([0, 1])([
        [0.4],
        [0.5],
        [0.6],
        [0.7],
    ]);

// init
auto linear = new Linear!(double, 2, 1);
auto optimizer = createOptimizer!SGD(linear);

// train
foreach (epoch; 0 .. 10_000)
{
    auto y = linear(data);
    auto diff = label - y;
    auto loss = mean(diff * diff);

    optimizer.resetGrads();
    loss.backward();
    optimizer.trainStep();
}

// result
auto y = linear(data);
writeln(y.value);
```
