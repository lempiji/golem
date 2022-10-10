import std;
import std.datetime.stopwatch;

void main()
{
    const t1 = measure!("dub build release", {
        spawnProcess(["dub", "build", "--build=release", "--compiler=ldc2", "-f"]).wait();
    });
    const t2 = measure!("release 64", {
        spawnProcess(["simple-linear", "64"]).wait();
    });
    const t3 = measure!("release 128", {
        spawnProcess(["simple-linear", "128"]).wait();
    });
    const t4 = measure!("release 256", {
        spawnProcess(["simple-linear", "256"]).wait();
    });
    const t5 = measure!("dub build release-nobounds", {
        spawnProcess(["dub", "build", "--build=release-nobounds", "--compiler=ldc2", "-f"]).wait();
    });
    const t6 = measure!("nobounds 64", {
        spawnProcess(["simple-linear", "64"]).wait();
    });
    const t7 = measure!("nobounds 128", {
        spawnProcess(["simple-linear", "128"]).wait();
    });
    const t8 = measure!("nobounds 256", {
        spawnProcess(["simple-linear", "256"]).wait();
    });

    writefln!"---- results ----"();
    writefln!"%s : %s"(t1.expand);
    writefln!"%s : %s"(t2.expand);
    writefln!"%s : %s"(t3.expand);
    writefln!"%s : %s"(t4.expand);
    writefln!"%s : %s"(t5.expand);
    writefln!"%s : %s"(t6.expand);
    writefln!"%s : %s"(t7.expand);
    writefln!"%s : %s"(t8.expand);
}

auto measure(string name, alias f)()
{
    StopWatch sw;
    sw.start();
    f();
    sw.stop();
    return tuple(name, sw.peek());
}
