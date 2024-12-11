
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_pkzb_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-9.346205891683829e-02, -8.371482262002407e-02, -4.959806172627839e-02, -1.810641598505199e-02, -1.095911360428545e-02, 7.469120677402707e-10, -1.905231215510539e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_pkzb_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.026712622893311e-01, -1.025093543119542e-01, -9.254539378426754e-02, -9.240668603323736e-02, -5.664537600732748e-02, -5.668890672673783e-02, -2.103988953676868e-02, -2.438789169143937e-03, -1.310473963828466e-02, -7.152685657482460e-02, -9.202443889536614e-11, 6.756354892214493e-09, -2.947886892506959e-19, -3.185343489683617e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_pkzb_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.391694708385569e-04, 2.771859809188080e-04, 1.382754904083679e-04, 5.969146803740697e-04, 1.193829360748139e-03, 5.969146803740697e-04, 1.796789276683559e-01, 3.593578553367119e-01, 1.796789276683559e-01, 3.135197842406391e+00, 8.349801861400495e+00, -3.120806951568205e+03, 1.681679230960683e+02, 3.363362362768979e+02, -9.766493913021875e+02, 5.022684028916403e-09, 3.267011171614460e-08, 2.405601967942859e-08, -3.361436188734302e-14, -3.043488421399815e-15, 3.859532227681246e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_pkzb_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-5.383358301103909e-08, 1.161967316132072e-08, -1.714312232586247e-76, -1.669949804750438e-76, -4.546418860831661e-69, -6.053550912339316e-69, 1.836240839087603e-03, 1.836239954192997e-03, 1.342090289942407e-12, 1.340380571299443e-12, 1.583504781024128e-13, -9.859161231622748e-10, 4.644388434465586e-24, 1.286287772348977e-23])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
