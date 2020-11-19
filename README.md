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
> found minimum at f(0.333329, 0.296200) = 0.544242
./matlab_rosenbrock
> found minimum at f(0.786469, 0.617630) = 0.045677
```