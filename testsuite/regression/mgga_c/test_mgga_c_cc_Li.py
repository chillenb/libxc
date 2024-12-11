
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_cc_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-9.348344939319045e-02, -8.371482262002408e-02, -4.959806172627839e-02, -1.808590345810658e-02, -1.095911360425487e-02, -6.777830421517112e-03, -1.131967991471019e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_cc_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.026428606104939e-01, -1.024807880673982e-01, -9.254539378426757e-02, -9.240668603323737e-02, -5.664537600732748e-02, -5.668890672673784e-02, -2.101619444369513e-02, -1.243107792619400e-01, -1.310473963818738e-02, -7.152742107535330e-02, -8.373134756108905e-03, -8.768732589292312e-03, -6.482554419365549e-05, -5.936010286578055e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_cc_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([3.891865759421425e-10, 7.783731518842850e-10, 3.891865759421425e-10, 9.555965868571372e-10, 1.911193173714274e-09, 9.555965868571372e-10, 1.286395797951029e-08, 2.572791595902057e-08, 1.286395797951029e-08, 2.128378562884594e+01, 4.256757125769189e+01, 2.128378562884594e+01, 6.394933884360409e+01, 1.278986776872082e+02, 6.394933884360409e+01, 3.615368343344791e-04, 7.230736686689582e-04, 3.615368343344791e-04, 1.717940799906916e+00, 3.435881599813832e+00, 1.717940799906916e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_cc_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.420478158362109e-11, -1.420478158362108e-11, -1.071641397466650e-50, -1.071641397466649e-50, -1.311306873672304e-47, -1.311306873672304e-47, -1.250636266914222e-05, -1.250636266913949e-05, -1.499379350903953e-13, -1.499379349870416e-13, -1.073377413639696e-08, -1.073377413639696e-08, -4.171701950315099e-10, -4.171701950315101e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
