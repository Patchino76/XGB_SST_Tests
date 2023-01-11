# Define the search-space dimensions. They must all have names!
from skopt.space import Real
from skopt import forest_minimize
from skopt.utils import use_named_args
dim1 = Real(name='foo', low=0, high=1)
dim2 = Real(name='bar', low=0.0, high=1.0)
dim3 = Real(name='baz', low=0.0, high=1.0)

# Gather the search-space dimensions in a list.
dimensions = [dim1, dim2, dim3]
print(dimensions)

# Define the objective function with named arguments
# and use this function-decorator to specify the
# search-space dimensions.
@use_named_args(dimensions=dimensions)
def my_objective_function(foo, bar, baz):
    return foo ** 2 + bar ** 4 + baz ** 8
# def my_objective_function(*data):
#     return data[0][0] ** 2 + data[0][1] ** 4 + data[0][2] ** 8


result = forest_minimize(func=my_objective_function,
                         dimensions=dimensions,
                         n_calls=20, base_estimator="ET",
                         random_state=4)

# Print the best-found results.
print("Best fitness:", result.fun)

print("Best parameters:", result.x)