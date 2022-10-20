^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package nlopt11
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.0.9 (2022-10-20)
------------------
* Fix build issues on no ROS environments

0.0.8 (2022-10-19)
------------------
* Support c-style function objects as objectives & constraints in opt11
* Remove opt11x

0.0.7 (2021-01-24)
------------------
* Add a variant of opt11[x]::optimize()
* Add Aithub action to build and run tests
* Add system dependency to libnlopt-cxx-dev

0.0.6 (2020-12-27)
------------------
* Make opt11x::add_xxx_mconstraint() callable without the explicit template param M

0.0.5 (2020-12-27)
------------------
* Expose std::vector versions of set_xxx() besides std::array versions in opt11x
* Minor refactors in examples

0.0.4 (2020-12-13)
------------------
* Eliminate cmake warnings
* Add more std::array-based methods to opt11x

0.0.3 (2020-12-11)
------------------
* Add opt11x, a faster version of opt11 but requiring dimension at compilation time
* Edit and add more examples to demonstrate opt11x and algorithm types

0.0.2 (2020-11-26)
------------------
* Explicitly compile as c++11
* Add more examples from a Umetani's book

0.0.1 (2020-11-24)
------------------
* Initial version tested on Ubuntu 18.04 and 20.04 with examples from nlopt, matlab and octave tutorials
* Contributors: Yoshito Okada
