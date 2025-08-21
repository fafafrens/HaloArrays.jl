# Haloarray

Distributed halo array utilities for Julia.

## Status
Work-in-progress. This repository currently contains scripts and utilities; it is not yet a registered Julia package.
In the file with the prefix test are some working example, like solving heat equation in parallel. 
Also there are some test to verify the logic and the corecntess, i even test the speed of some routine. 
The main goal was to have an array for distributed computing that simply work. 
Currently the main logic of halo excange is implementet in a way that can work with normal array or 
possible GPU array. 
The boundary condition are implemted but i am not happy. 
The broadcast looks solid works in place and in out of place, this i am really happy i do not know how 
i managed to do it. 
I define also an array that take multiple haloarray with nametuple but it beahves like an array, with the broadcast and ecc. 

I define a gather function and mapreduce of the total array. 
The mapreduce of some silce is under construction since is hard. 
I have wrote some function for printing into hdf5 in parallel or 
on after a gather on root rank. 


## Getting started
- Clone the repository:
  ```bash
  git clone https://github.com/fafafrens/Haloarray.git
  cd Haloarray
  ```
- Use or explore the Julia scripts directly. For example, from a Julia REPL in this directory you can include specific files:
  ```julia
  include("haloarray.jl")
  # or
  include("haloarrays.jl")
  ```

## Contributing
Issues and pull requests are welcome. Please open an issue to discuss major changes.
please help me :-)

## License
MIT — see [LICENSE](./LICENSE).

