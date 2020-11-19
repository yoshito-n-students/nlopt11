# nlopt11
nlplot::opt in flavor of c++11

## How to build & run
First time
```
git clone git@github.com:yoshito-n-students/nlopt11.git
cd nlopt11
mkdir build
cd build
cmake ..
make
```

Second time or later
```
cd build
make
```

Solve example problems
```
cd build
./nlopt_tutorial
> found minimum at f(0.333333, 0.296296) = 0.544331
./matlab_rosenbrock
> found minimum at f(0.786469, 0.617630) = 0.045677
./octave_qp5
> found minimum at f(-1.717138, 1.595703, 1.827256, -0.763661, -0.763627) = -0.053950
...
```