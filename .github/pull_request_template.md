# PR Checklist

* [ ] Project compiles locally and on CI without error
* [ ] Project passes all regression tests
* [ ] Unit and integration tests added for new features
* [ ] New tests are integrated into the project's GTest framework and are called by test_all
* [ ] New files added to CMake
* [ ] Every header has include guards
* [ ] Code adheres to project naming conventions
* [ ] Every class with a destructor has a copy/move constructor (may be default or deleted)
* [ ] Every class with a virtual function has a virtual destructor
* [ ] References are used where possible to avoid unnecessary copies