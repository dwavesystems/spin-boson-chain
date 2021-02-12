#!/usr/bin/env python
r"""This script runs several tests on the :mod:`sbc.system` module."""



#####################################
## Load libraries/packages/modules ##
#####################################

# Import class representing time-dependent scalar model parameters.
from sbc.scalar import Scalar

# Module to test.
from sbc import system



############################
## Authorship information ##
############################

__author__ = "Matthew Fitzpatrick"
__copyright__ = "Copyright 2021"
__credits__ = ["Matthew Fitzpatrick"]
__maintainer__ = "Matthew Fitzpatrick"
__email__ = "mfitzpatrick@dwavesys.com"
__status__ = "Non-Production"



#########################
## Main body of script ##
#########################

# system.Model test #1.
print("system.Model test #1")
print("====================")

print("Constructing a system model parameter set.\n")
model = system.Model()
print("Print attributes of object:")
print("    z_fields =", model.z_fields)
print("    x_fields =", model.x_fields)
print("    zz_couplers =", model.zz_couplers)
print("    L =", model.L)
print()
print("Evaluate z_fields[0] at t=0:")
print("    z_fields[0](t=0) =", model.z_fields[0].eval(0))
print("Evaluate z_fields[0] at t=2.0:")
print("    z_fields[0](t=2.0) =", model.z_fields[0].eval(2.0))
print("Evaluate x_fields[0] at t=0:")
print("    x_fields[0](t=0) =", model.x_fields[0].eval(0))
print("Evaluate x_fields[0] at t=2.0:")
print("    x_fields[0](t=2.0) =", model.x_fields[0].eval(2.0))
print("\n\n")



# system.Model test #2.
print("system.Model test #2")
print("====================")

def linear_fn(t, a, b):
    return a*t+b

const_scalar = 2.5

func_kwargs_1 = {"a": 2.0, "b": -1.0}
func_kwargs_2 = {"a": 4.0, "b": -2.0}

model_param_1 = Scalar(linear_fn, func_kwargs_1)
model_param_2 = Scalar(linear_fn, func_kwargs_2)

z_fields = [model_param_1, model_param_2, const_scalar]

print("Constructing a system model parameter set.\n")
model = system.Model(z_fields=z_fields)
print("Print attributes of object:")
print("    z_fields =", model.z_fields)
print("    x_fields =", model.x_fields)
print("    zz_couplers =", model.zz_couplers)
print("    L =", model.L)
print()
print("Evaluate z_fields[0] at t=2.0:")
print("    z_fields[0](t=2.0) =", model.z_fields[0].eval(2.0))
print("Evaluate z_fields[1] at t=2.0:")
print("    z_fields[1](t=2.0) =", model.z_fields[1].eval(2.0))
print("Evaluate z_fields[2] at t=2.0:")
print("    z_fields[2](t=2.0) =", model.z_fields[2].eval(2.0))
print("Evaluate x_fields[0] at t=2.0:")
print("    x_fields[0](t=2.0) =", model.x_fields[0].eval(2.0))
print("Evaluate x_fields[1] at t=2.0:")
print("    x_fields[1](t=2.0) =", model.x_fields[1].eval(2.0))
print("Evaluate x_fields[2] at t=2.0:")
print("    x_fields[2](t=2.0) =", model.x_fields[2].eval(2.0))
print("Evaluate zz_couplers[0] at t=2.0:")
print("    zz_couplers[0](t=2.0) =", model.zz_couplers[0].eval(2.0))
print("Evaluate zz_couplers[1] at t=2.0:")
print("    zz_couplers[1](t=2.0) =", model.zz_couplers[1].eval(2.0))
print("\n\n")



# system.Model test #3.
print("system.Model test #3")
print("====================")

def linear_fn(t, a, b):
    return a*t+b

const_scalar = 2.5

func_kwargs_1 = {"a": 2.0, "b": -1.0}
func_kwargs_2 = {"a": 4.0, "b": -2.0}

model_param_1 = Scalar(linear_fn, func_kwargs_1)
model_param_2 = Scalar(linear_fn, func_kwargs_2)

x_fields = [model_param_1, model_param_2, const_scalar]

print("Constructing a system model parameter set.\n")
model = system.Model(x_fields=x_fields)
print("Print attributes of object:")
print("    z_fields =", model.z_fields)
print("    x_fields =", model.x_fields)
print("    zz_couplers =", model.zz_couplers)
print("    L =", model.L)
print()
print("Evaluate z_fields[0] at t=2.0:")
print("    z_fields[0](t=2.0) =", model.z_fields[0].eval(2.0))
print("Evaluate z_fields[1] at t=2.0:")
print("    z_fields[1](t=2.0) =", model.z_fields[1].eval(2.0))
print("Evaluate z_fields[2] at t=2.0:")
print("    z_fields[2](t=2.0) =", model.z_fields[2].eval(2.0))
print("Evaluate x_fields[0] at t=2.0:")
print("    x_fields[0](t=2.0) =", model.x_fields[0].eval(2.0))
print("Evaluate x_fields[1] at t=2.0:")
print("    x_fields[1](t=2.0) =", model.x_fields[1].eval(2.0))
print("Evaluate x_fields[2] at t=2.0:")
print("    x_fields[2](t=2.0) =", model.x_fields[2].eval(2.0))
print("Evaluate zz_couplers[0] at t=2.0:")
print("    zz_couplers[0](t=2.0) =", model.zz_couplers[0].eval(2.0))
print("Evaluate zz_couplers[1] at t=2.0:")
print("    zz_couplers[1](t=2.0) =", model.zz_couplers[1].eval(2.0))
print("\n\n")



# system.Model test #4.
print("system.Model test #4")
print("====================")

def linear_fn(t, a, b):
    return a*t+b

const_scalar = 2.5

func_kwargs_1 = {"a": 2.0, "b": -1.0}

model_param_1 = Scalar(linear_fn, func_kwargs_1)

zz_couplers = [model_param_1, const_scalar]

print("Constructing a system model parameter set.\n")
model = system.Model(zz_couplers=zz_couplers)
print("Print attributes of object:")
print("    z_fields =", model.z_fields)
print("    x_fields =", model.x_fields)
print("    zz_couplers =", model.zz_couplers)
print("    L =", model.L)
print()
print("Evaluate z_fields[0] at t=2.0:")
print("    z_fields[0](t=2.0) =", model.z_fields[0].eval(2.0))
print("Evaluate z_fields[1] at t=2.0:")
print("    z_fields[1](t=2.0) =", model.z_fields[1].eval(2.0))
print("Evaluate z_fields[2] at t=2.0:")
print("    z_fields[2](t=2.0) =", model.z_fields[2].eval(2.0))
print("Evaluate x_fields[0] at t=2.0:")
print("    x_fields[0](t=2.0) =", model.x_fields[0].eval(2.0))
print("Evaluate x_fields[1] at t=2.0:")
print("    x_fields[1](t=2.0) =", model.x_fields[1].eval(2.0))
print("Evaluate x_fields[2] at t=2.0:")
print("    x_fields[2](t=2.0) =", model.x_fields[2].eval(2.0))
print("Evaluate zz_couplers[0] at t=2.0:")
print("    zz_couplers[0](t=2.0) =", model.zz_couplers[0].eval(2.0))
print("Evaluate zz_couplers[1] at t=2.0:")
print("    zz_couplers[1](t=2.0) =", model.zz_couplers[1].eval(2.0))
print("\n\n")



# system.Model test #5.
print("system.Model test #5")
print("====================")

def linear_fn(t, a, b):
    return a*t+b

const_scalar = 2.5

func_kwargs_1 = {"a": 2.0, "b": -1.0}
func_kwargs_2 = {"a": 4.0, "b": -2.0}

model_param_1 = Scalar(linear_fn, func_kwargs_1)
model_param_2 = Scalar(linear_fn, func_kwargs_2)

z_fields = [model_param_1, model_param_2, const_scalar]

print("Constructing a system model parameter set.\n")
model = system.Model(z_fields=z_fields, x_fields=[], zz_couplers=[])
print("Print attributes of object:")
print("    z_fields =", model.z_fields)
print("    x_fields =", model.x_fields)
print("    zz_couplers =", model.zz_couplers)
print("    L =", model.L)
print()
print("Evaluate z_fields[0] at t=2.0:")
print("    z_fields[0](t=2.0) =", model.z_fields[0].eval(2.0))
print("Evaluate z_fields[1] at t=2.0:")
print("    z_fields[1](t=2.0) =", model.z_fields[1].eval(2.0))
print("Evaluate z_fields[2] at t=2.0:")
print("    z_fields[2](t=2.0) =", model.z_fields[2].eval(2.0))
print("Evaluate x_fields[0] at t=2.0:")
print("    x_fields[0](t=2.0) =", model.x_fields[0].eval(2.0))
print("Evaluate x_fields[1] at t=2.0:")
print("    x_fields[1](t=2.0) =", model.x_fields[1].eval(2.0))
print("Evaluate x_fields[2] at t=2.0:")
print("    x_fields[2](t=2.0) =", model.x_fields[2].eval(2.0))
print("Evaluate zz_couplers[0] at t=2.0:")
print("    zz_couplers[0](t=2.0) =", model.zz_couplers[0].eval(2.0))
print("Evaluate zz_couplers[1] at t=2.0:")
print("    zz_couplers[1](t=2.0) =", model.zz_couplers[1].eval(2.0))
print("\n\n")



# system.Model test #6.
print("system.Model test #6")
print("====================")

def linear_fn(t, a, b):
    return a*t+b

def quad_fn(t, a, b):
    return a * t * t + b

const_scalar_1 = 2.5
const_scalar_2 = -2.5

func_kwargs_1 = {"a": 2.0, "b": -1.0}
func_kwargs_2 = {"a": 4.0, "b": -2.0}
func_kwargs_3 = {"a": -2.0, "b": 1.0}
func_kwargs_4 = {"a": -4.0, "b": 2.0}

model_param_1 = Scalar(linear_fn, func_kwargs_1)
model_param_2 = Scalar(linear_fn, func_kwargs_2)
model_param_3 = Scalar(quad_fn, func_kwargs_3)
model_param_4 = Scalar(quad_fn, func_kwargs_4)

z_fields = [model_param_1, model_param_2, const_scalar_1]
x_fields = [model_param_3, model_param_4, const_scalar_2]

print("Constructing a system model parameter set.\n")
model = system.Model(z_fields=z_fields, x_fields=x_fields)
print("Print attributes of object:")
print("    z_fields =", model.z_fields)
print("    x_fields =", model.x_fields)
print("    zz_couplers =", model.zz_couplers)
print("    L =", model.L)
print()
print("Evaluate z_fields[0] at t=2.0:")
print("    z_fields[0](t=2.0) =", model.z_fields[0].eval(2.0))
print("Evaluate z_fields[1] at t=2.0:")
print("    z_fields[1](t=2.0) =", model.z_fields[1].eval(2.0))
print("Evaluate z_fields[2] at t=2.0:")
print("    z_fields[2](t=2.0) =", model.z_fields[2].eval(2.0))
print("Evaluate x_fields[0] at t=2.0:")
print("    x_fields[0](t=2.0) =", model.x_fields[0].eval(2.0))
print("Evaluate x_fields[1] at t=2.0:")
print("    x_fields[1](t=2.0) =", model.x_fields[1].eval(2.0))
print("Evaluate x_fields[2] at t=2.0:")
print("    x_fields[2](t=2.0) =", model.x_fields[2].eval(2.0))
print("Evaluate zz_couplers[0] at t=2.0:")
print("    zz_couplers[0](t=2.0) =", model.zz_couplers[0].eval(2.0))
print("Evaluate zz_couplers[1] at t=2.0:")
print("    zz_couplers[1](t=2.0) =", model.zz_couplers[1].eval(2.0))
print("\n\n")



# system.Model test #7.
print("system.Model test #7")
print("====================")

def linear_fn(t, a, b):
    return a*t+b

def quad_fn(t, a, b):
    return a * t * t + b

const_scalar_1 = 2.5
const_scalar_2 = -2.5

func_kwargs_1 = {"a": 2.0, "b": -1.0}
func_kwargs_2 = {"a": 4.0, "b": -2.0}
func_kwargs_3 = {"a": -2.0, "b": 1.0}

model_param_1 = Scalar(linear_fn, func_kwargs_1)
model_param_2 = Scalar(linear_fn, func_kwargs_2)
model_param_3 = Scalar(quad_fn, func_kwargs_3)

z_fields = [model_param_1, model_param_2, const_scalar_1]
x_fields = [model_param_3, const_scalar_2]

print("Constructing a system model parameter set; Expecting an IndexError "
      "exception.\n")
try:
    model = system.Model(z_fields=z_fields, x_fields=x_fields)
except IndexError as e:
    print(e)
    print("\n\n")



# system.Model test #8.
print("system.Model test #8")
print("====================")

def linear_fn(t, a, b):
    return a*t+b

def quad_fn(t, a, b):
    return a * t * t + b

const_scalar_1 = 2.5
const_scalar_2 = -2.5

func_kwargs_1 = {"a": 2.0, "b": -1.0}
func_kwargs_2 = {"a": 4.0, "b": -2.0}
func_kwargs_3 = {"a": -2.0, "b": 1.0}

model_param_1 = Scalar(linear_fn, func_kwargs_1)
model_param_2 = Scalar(linear_fn, func_kwargs_2)
model_param_3 = Scalar(quad_fn, func_kwargs_3)

z_fields = [model_param_1, model_param_2, const_scalar_1]
zz_couplers = [model_param_3, const_scalar_2]

print("Constructing a system model parameter set.\n")
model = system.Model(z_fields=z_fields, zz_couplers=zz_couplers)
print("Print attributes of object:")
print("    z_fields =", model.z_fields)
print("    x_fields =", model.x_fields)
print("    zz_couplers =", model.zz_couplers)
print("    L =", model.L)
print()
print("Evaluate z_fields[0] at t=2.0:")
print("    z_fields[0](t=2.0) =", model.z_fields[0].eval(2.0))
print("Evaluate z_fields[1] at t=2.0:")
print("    z_fields[1](t=2.0) =", model.z_fields[1].eval(2.0))
print("Evaluate z_fields[2] at t=2.0:")
print("    z_fields[2](t=2.0) =", model.z_fields[2].eval(2.0))
print("Evaluate x_fields[0] at t=2.0:")
print("    x_fields[0](t=2.0) =", model.x_fields[0].eval(2.0))
print("Evaluate x_fields[1] at t=2.0:")
print("    x_fields[1](t=2.0) =", model.x_fields[1].eval(2.0))
print("Evaluate x_fields[2] at t=2.0:")
print("    x_fields[2](t=2.0) =", model.x_fields[2].eval(2.0))
print("Evaluate zz_couplers[0] at t=2.0:")
print("    zz_couplers[0](t=2.0) =", model.zz_couplers[0].eval(2.0))
print("Evaluate zz_couplers[1] at t=2.0:")
print("    zz_couplers[1](t=2.0) =", model.zz_couplers[1].eval(2.0))
print("\n\n")



# system.Model test #9.
print("system.Model test #9")
print("====================")

def linear_fn(t, a, b):
    return a*t+b

def quad_fn(t, a, b):
    return a * t * t + b

const_scalar_1 = 2.5
const_scalar_2 = -2.5

func_kwargs_1 = {"a": 2.0, "b": -1.0}
func_kwargs_2 = {"a": 4.0, "b": -2.0}

model_param_1 = Scalar(linear_fn, func_kwargs_1)
model_param_2 = Scalar(quad_fn, func_kwargs_2)

x_fields = [model_param_1, const_scalar_1]
zz_couplers = [model_param_2, const_scalar_2]

print("Constructing a system model parameter set; Expecting an IndexError "
      "exception.\n")
try:
    model = system.Model(x_fields=x_fields, zz_couplers=zz_couplers)
except IndexError as e:
    print(e)
    print("\n\n")



# system.Model test #10.
print("system.Model test #10")
print("====================")

def linear_fn(t, a, b):
    return a*t+b

def quad_fn(t, a, b):
    return a * t * t + b

def cubic_fn(t, a, b):
    return a * t * t * t + b

const_scalar_1 = 2.5
const_scalar_2 = -2.5

func_kwargs_1 = {"a": 2.0, "b": -1.0}
func_kwargs_2 = {"a": 4.0, "b": -2.0}
func_kwargs_3 = {"a": -2.0, "b": 1.0}
func_kwargs_4 = {"a": -4.0, "b": 2.0}
func_kwargs_5 = {"a": 10, "b": 1}
func_kwargs_6 = {"a": -10, "b": -1}

model_param_1 = Scalar(linear_fn, func_kwargs_1)
model_param_2 = Scalar(linear_fn, func_kwargs_2)
model_param_3 = Scalar(quad_fn, func_kwargs_3)
model_param_4 = Scalar(quad_fn, func_kwargs_4)
model_param_5 = Scalar(cubic_fn, func_kwargs_5)
model_param_6 = Scalar(cubic_fn, func_kwargs_6)

z_fields = [model_param_1, model_param_2, const_scalar_1]
x_fields = [model_param_3, model_param_4, const_scalar_2]
zz_couplers = [model_param_5, model_param_6]

print("Constructing a system model parameter set.\n")
model = system.Model(z_fields=z_fields,
                     x_fields=x_fields,
                     zz_couplers=zz_couplers)
print("Print attributes of object:")
print("    z_fields =", model.z_fields)
print("    x_fields =", model.x_fields)
print("    zz_couplers =", model.zz_couplers)
print("    L =", model.L)
print()
print("Evaluate z_fields[0] at t=2.0:")
print("    z_fields[0](t=2.0) =", model.z_fields[0].eval(2.0))
print("Evaluate z_fields[1] at t=2.0:")
print("    z_fields[1](t=2.0) =", model.z_fields[1].eval(2.0))
print("Evaluate z_fields[2] at t=2.0:")
print("    z_fields[2](t=2.0) =", model.z_fields[2].eval(2.0))
print("Evaluate x_fields[0] at t=2.0:")
print("    x_fields[0](t=2.0) =", model.x_fields[0].eval(2.0))
print("Evaluate x_fields[1] at t=2.0:")
print("    x_fields[1](t=2.0) =", model.x_fields[1].eval(2.0))
print("Evaluate x_fields[2] at t=2.0:")
print("    x_fields[2](t=2.0) =", model.x_fields[2].eval(2.0))
print("Evaluate zz_couplers[0] at t=2.0:")
print("    zz_couplers[0](t=2.0) =", model.zz_couplers[0].eval(2.0))
print("Evaluate zz_couplers[1] at t=2.0:")
print("    zz_couplers[1](t=2.0) =", model.zz_couplers[1].eval(2.0))
print("\n\n")



# system.Model test #11.
print("system.Model test #11")
print("====================")

def linear_fn(t, a, b):
    return a*t+b

def quad_fn(t, a, b):
    return a * t * t + b

def cubic_fn(t, a, b):
    return a * t * t * t + b

const_scalar_1 = 2.5

func_kwargs_1 = {"a": 2.0, "b": -1.0}
func_kwargs_2 = {"a": 4.0, "b": -2.0}
func_kwargs_3 = {"a": -2.0, "b": 1.0}
func_kwargs_4 = {"a": -4.0, "b": 2.0}
func_kwargs_5 = {"a": 10, "b": 1}
func_kwargs_6 = {"a": -10, "b": -1}

model_param_1 = Scalar(linear_fn, func_kwargs_1)
model_param_2 = Scalar(linear_fn, func_kwargs_2)
model_param_3 = Scalar(quad_fn, func_kwargs_3)
model_param_4 = Scalar(quad_fn, func_kwargs_4)
model_param_5 = Scalar(cubic_fn, func_kwargs_5)
model_param_6 = Scalar(cubic_fn, func_kwargs_6)

z_fields = [model_param_1, model_param_2, const_scalar_1]
x_fields = [model_param_3, model_param_4]
zz_couplers = [model_param_5, model_param_6]

print("Constructing a system model parameter set; Expecting an IndexError "
      "exception.\n")
try:
    model = system.Model(z_fields=z_fields,
                         x_fields=x_fields,
                         zz_couplers=zz_couplers)
except IndexError as e:
    print(e)
    print("\n\n")



# system.Model test #12.
print("system.Model test #12")
print("====================")

def linear_fn(t, a, b):
    return a*t+b

def quad_fn(t, a, b):
    return a * t * t + b

def cubic_fn(t, a, b):
    return a * t * t * t + b

const_scalar_1 = 2.5
const_scalar_2 = -2.5

func_kwargs_1 = {"a": 2.0, "b": -1.0}
func_kwargs_2 = {"a": 4.0, "b": -2.0}
func_kwargs_3 = {"a": -2.0, "b": 1.0}
func_kwargs_4 = {"a": -4.0, "b": 2.0}
func_kwargs_5 = {"a": 10, "b": 1}
func_kwargs_6 = {"a": -10, "b": -1}

model_param_1 = Scalar(linear_fn, func_kwargs_1)
model_param_2 = Scalar(linear_fn, func_kwargs_2)
model_param_3 = Scalar(quad_fn, func_kwargs_3)
model_param_4 = Scalar(quad_fn, func_kwargs_4)
model_param_5 = Scalar(cubic_fn, func_kwargs_5)
model_param_6 = Scalar(cubic_fn, func_kwargs_6)

z_fields = [model_param_1, model_param_2, const_scalar_1,
            model_param_3, const_scalar_2]
x_fields = [model_param_3, model_param_4, const_scalar_2,
            const_scalar_2, model_param_3]
zz_couplers = [model_param_5, model_param_6, model_param_6, model_param_6]

print("Constructing a system model parameter set.\n")
model = system.Model(z_fields=z_fields,
                     x_fields=x_fields,
                     zz_couplers=zz_couplers)
print("Print attributes of object:")
print("    z_fields =", model.z_fields)
print("    x_fields =", model.x_fields)
print("    zz_couplers =", model.zz_couplers)
print("    _map_btwn_site_indices_and_unique_x_fields =",
      model._map_btwn_site_indices_and_unique_x_fields)
print("    L =", model.L)
print()
print("Evaluate z_fields[0] at t=2.0:")
print("    z_fields[0](t=2.0) =", model.z_fields[0].eval(2.0))
print("Evaluate z_fields[1] at t=2.0:")
print("    z_fields[1](t=2.0) =", model.z_fields[1].eval(2.0))
print("Evaluate z_fields[2] at t=2.0:")
print("    z_fields[2](t=2.0) =", model.z_fields[2].eval(2.0))
print("Evaluate z_fields[3] at t=2.0:")
print("    z_fields[3](t=2.0) =", model.z_fields[3].eval(2.0))
print("Evaluate z_fields[4] at t=2.0:")
print("    z_fields[4](t=2.0) =", model.z_fields[4].eval(2.0))
print("Evaluate z_fields[0] at t=2.0:")
print("    x_fields[0](t=2.0) =", model.x_fields[0].eval(2.0))
print("Evaluate x_fields[1] at t=2.0:")
print("    x_fields[1](t=2.0) =", model.x_fields[1].eval(2.0))
print("Evaluate x_fields[2] at t=2.0:")
print("    x_fields[2](t=2.0) =", model.x_fields[2].eval(2.0))
print("Evaluate x_fields[3] at t=2.0:")
print("    x_fields[3](t=2.0) =", model.x_fields[3].eval(2.0))
print("Evaluate x_fields[4] at t=2.0:")
print("    x_fields[4](t=2.0) =", model.x_fields[4].eval(2.0))
print("Evaluate zz_couplers[0] at t=2.0:")
print("    zz_couplers[0](t=2.0) =", model.zz_couplers[0].eval(2.0))
print("Evaluate zz_couplers[1] at t=2.0:")
print("    zz_couplers[1](t=2.0) =", model.zz_couplers[1].eval(2.0))
print("Evaluate zz_couplers[2] at t=2.0:")
print("    zz_couplers[2](t=2.0) =", model.zz_couplers[2].eval(2.0))
print("Evaluate zz_couplers[3] at t=2.0:")
print("    zz_couplers[3](t=2.0) =", model.zz_couplers[3].eval(2.0))
print("\n\n")



# system.Model test #13.
print("system.Model test #13")
print("====================")

def linear_fn(t, a, b):
    return a*t+b

def quad_fn(t, a, b):
    return a * t * t + b

def cubic_fn(t, a, b):
    return a * t * t * t + b

const_scalar_1 = 2.5
const_scalar_2 = -2.5

func_kwargs_1 = {"a": 2.0, "b": -1.0}
func_kwargs_2 = {"a": 4.0, "b": -2.0}
func_kwargs_3 = {"a": -2.0, "b": 1.0}
func_kwargs_4 = {"a": -4.0, "b": 2.0}
func_kwargs_5 = {"a": 10, "b": 1}
func_kwargs_6 = {"a": -10, "b": -1}

model_param_1 = Scalar(linear_fn, func_kwargs_1)
model_param_2 = Scalar(linear_fn, func_kwargs_2)
model_param_3 = Scalar(quad_fn, func_kwargs_3)
model_param_4 = Scalar(quad_fn, func_kwargs_4)
model_param_5 = Scalar(cubic_fn, func_kwargs_5)
model_param_6 = Scalar(cubic_fn, func_kwargs_6)

z_fields = [model_param_1, model_param_2, const_scalar_1,
            model_param_3, model_param_4]
zz_couplers = [model_param_5, model_param_6, model_param_6, const_scalar_2]

print("Constructing a system model parameter set.\n")
model = system.Model(z_fields=z_fields, zz_couplers=zz_couplers)
print("Print attributes of object:")
print("    z_fields =", model.z_fields)
print("    x_fields =", model.x_fields)
print("    zz_couplers =", model.zz_couplers)
print("    _map_btwn_site_indices_and_unique_x_fields =",
      model._map_btwn_site_indices_and_unique_x_fields)
print("    L =", model.L)
print()
print("Evaluate z_fields[0] at t=2.0:")
print("    z_fields[0](t=2.0) =", model.z_fields[0].eval(2.0))
print("Evaluate z_fields[1] at t=2.0:")
print("    z_fields[1](t=2.0) =", model.z_fields[1].eval(2.0))
print("Evaluate z_fields[2] at t=2.0:")
print("    z_fields[2](t=2.0) =", model.z_fields[2].eval(2.0))
print("Evaluate z_fields[3] at t=2.0:")
print("    z_fields[3](t=2.0) =", model.z_fields[3].eval(2.0))
print("Evaluate z_fields[4] at t=2.0:")
print("    z_fields[4](t=2.0) =", model.z_fields[4].eval(2.0))
print("Evaluate z_fields[0] at t=2.0:")
print("    x_fields[0](t=2.0) =", model.x_fields[0].eval(2.0))
print("Evaluate x_fields[1] at t=2.0:")
print("    x_fields[1](t=2.0) =", model.x_fields[1].eval(2.0))
print("Evaluate x_fields[2] at t=2.0:")
print("    x_fields[2](t=2.0) =", model.x_fields[2].eval(2.0))
print("Evaluate x_fields[3] at t=2.0:")
print("    x_fields[3](t=2.0) =", model.x_fields[3].eval(2.0))
print("Evaluate x_fields[4] at t=2.0:")
print("    x_fields[4](t=2.0) =", model.x_fields[4].eval(2.0))
print("Evaluate zz_couplers[0] at t=2.0:")
print("    zz_couplers[0](t=2.0) =", model.zz_couplers[0].eval(2.0))
print("Evaluate zz_couplers[1] at t=2.0:")
print("    zz_couplers[1](t=2.0) =", model.zz_couplers[1].eval(2.0))
print("Evaluate zz_couplers[2] at t=2.0:")
print("    zz_couplers[2](t=2.0) =", model.zz_couplers[2].eval(2.0))
print("Evaluate zz_couplers[3] at t=2.0:")
print("    zz_couplers[3](t=2.0) =", model.zz_couplers[3].eval(2.0))
print("\n\n")



# system.Model test #14.
print("system.Model test #14")
print("====================")

def linear_fn(t, a, b):
    return a*t+b

def quad_fn(t, a, b):
    return a * t * t + b

def cubic_fn(t, a, b):
    return a * t * t * t + b

const_scalar_1 = 2.5
const_scalar_2 = -2.5

func_kwargs_1 = {"a": 2.0, "b": -1.0}
func_kwargs_2 = {"a": 4.0, "b": -2.0}
func_kwargs_3 = {"a": -2.0, "b": 1.0}
func_kwargs_4 = {"a": -4.0, "b": 2.0}
func_kwargs_5 = {"a": 10, "b": 1}
func_kwargs_6 = {"a": -10, "b": -1}

model_param_1 = Scalar(linear_fn, func_kwargs_1)
model_param_2 = Scalar(linear_fn, func_kwargs_2)
model_param_3 = Scalar(quad_fn, func_kwargs_3)
model_param_4 = Scalar(quad_fn, func_kwargs_4)
model_param_5 = Scalar(cubic_fn, func_kwargs_5)
model_param_6 = Scalar(cubic_fn, func_kwargs_6)

z_fields = [model_param_1, model_param_2, const_scalar_1,
            model_param_3, const_scalar_2]
x_fields = [const_scalar_2, model_param_3, model_param_4,
            model_param_3, model_param_4]
zz_couplers = [model_param_5, model_param_6, model_param_6, model_param_6]

print("Constructing a system model parameter set.\n")
model = system.Model(z_fields=z_fields,
                     x_fields=x_fields,
                     zz_couplers=zz_couplers)
print("Print attributes of object:")
print("    z_fields =", model.z_fields)
print("    x_fields =", model.x_fields)
print("    zz_couplers =", model.zz_couplers)
print("    _map_btwn_site_indices_and_unique_x_fields =",
      model._map_btwn_site_indices_and_unique_x_fields)
print("    L =", model.L)
print()
print("Evaluate z_fields[0] at t=2.0:")
print("    z_fields[0](t=2.0) =", model.z_fields[0].eval(2.0))
print("Evaluate z_fields[1] at t=2.0:")
print("    z_fields[1](t=2.0) =", model.z_fields[1].eval(2.0))
print("Evaluate z_fields[2] at t=2.0:")
print("    z_fields[2](t=2.0) =", model.z_fields[2].eval(2.0))
print("Evaluate z_fields[3] at t=2.0:")
print("    z_fields[3](t=2.0) =", model.z_fields[3].eval(2.0))
print("Evaluate z_fields[4] at t=2.0:")
print("    z_fields[4](t=2.0) =", model.z_fields[4].eval(2.0))
print("Evaluate z_fields[0] at t=2.0:")
print("    x_fields[0](t=2.0) =", model.x_fields[0].eval(2.0))
print("Evaluate x_fields[1] at t=2.0:")
print("    x_fields[1](t=2.0) =", model.x_fields[1].eval(2.0))
print("Evaluate x_fields[2] at t=2.0:")
print("    x_fields[2](t=2.0) =", model.x_fields[2].eval(2.0))
print("Evaluate x_fields[3] at t=2.0:")
print("    x_fields[3](t=2.0) =", model.x_fields[3].eval(2.0))
print("Evaluate x_fields[4] at t=2.0:")
print("    x_fields[4](t=2.0) =", model.x_fields[4].eval(2.0))
print("Evaluate zz_couplers[0] at t=2.0:")
print("    zz_couplers[0](t=2.0) =", model.zz_couplers[0].eval(2.0))
print("Evaluate zz_couplers[1] at t=2.0:")
print("    zz_couplers[1](t=2.0) =", model.zz_couplers[1].eval(2.0))
print("Evaluate zz_couplers[2] at t=2.0:")
print("    zz_couplers[2](t=2.0) =", model.zz_couplers[2].eval(2.0))
print("Evaluate zz_couplers[3] at t=2.0:")
print("    zz_couplers[3](t=2.0) =", model.zz_couplers[3].eval(2.0))
