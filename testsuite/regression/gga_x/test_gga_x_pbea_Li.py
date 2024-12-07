
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pbea_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbea", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.793311367606394e+00, -1.280896283382549e+00, -3.969557940301007e-01, -1.599897083384555e-01, -7.829723275141584e-02, -2.034675949209563e-02, -3.837619917378862e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pbea_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbea", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.246582847949993e+00, -2.248707165724058e+00, -1.528151087792309e+00, -1.529510654707136e+00, -4.083654744243794e-01, -4.084507295897952e-01, -2.054428080874089e-01, -2.563737708995404e-02, -7.965661561625334e-02, -8.289061826506319e-04, -2.692756939063569e-02, -2.674617251262807e-02, -5.538570384016349e-04, -3.937907436785811e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pbea_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbea", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.458782229522660e-04, 0.000000000000000e+00, -2.450465102509665e-04, -9.432179683742399e-04, 0.000000000000000e+00, -9.402556686822453e-04, -5.864694002704488e-02, 0.000000000000000e+00, -5.850885900217612e-02, -3.863049888623784e+00, 0.000000000000000e+00, -2.553905241627678e+00, -5.413625060894128e+01, 0.000000000000000e+00, -2.871550720825923e+02, -2.463737799309921e+00, 0.000000000000000e+00, -2.355148309167455e+00, -3.323299297585597e+02, 0.000000000000000e+00, -6.056277567728895e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
