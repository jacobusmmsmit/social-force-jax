# social-force-diffrax
An implementation of a simple Social Force model in JAX + Diffrax as well as a differentiable density-based optimisation of the parameters with Optax or numpyro.

To run the code in this repo, install the poetry package manager, navigate to the folder, and run `poetry install`. This will install all dependencies.

Note: The local file imports are kinda messed up for this repo, you may need to play around with the path to get it working properly.

### Known issues:
* The wall force sometimes lets pedestrians escape (see animation), oops.
