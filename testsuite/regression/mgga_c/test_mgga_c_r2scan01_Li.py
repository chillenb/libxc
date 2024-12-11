
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_r2scan01_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.100636804569064e-02, -5.629325295468333e-02, -6.498879409859544e-02, -2.344946743553334e-03, -1.518603161347360e-02, -2.549715731629722e-04, -2.994175370304462e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_r2scan01_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.343962437785070e-02, -1.329165339925406e-02, 1.051458023500476e-03, 1.263384416414214e-03, -6.820947134485039e-02, -6.826684027960464e-02, 1.780111884975611e-03, -1.571689225129233e-01, -1.229191849435341e-02, -5.721243928703525e-02, -2.696369432040480e-03, -2.703240649110413e-03, -3.195629941120045e-08, -1.746615367446432e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_r2scan01_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [7.106755940538392e-05, 1.421351188107678e-04, 7.106755940538392e-05, 5.582156495126489e-04, 1.116431299025298e-03, 5.582156495126489e-04, 2.078849185090517e-01, 4.157698370181034e-01, 2.078849185090517e-01, 3.519442243709761e+00, 7.038884487419524e+00, 3.519442243709761e+00, 1.307226074455604e+02, 2.614452148911208e+02, 1.307226074455604e+02, 9.993473452485976e-01, 1.998694690497195e+00, 9.993473452485976e-01, 4.761177366037663e-04, 9.522354732075332e-04, 4.761177366037663e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_r2scan01_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-4.266282960320483e-03, -4.266282960320482e-03, -9.116363258336245e-03, -9.116363258336242e-03, -1.278855002579240e-03, -1.278855002579239e-03, -1.209296596158873e-01, -1.209296596158606e-01, -4.912667931257945e-02, -4.912667927288967e-02, -2.883280201538325e-05, -2.883280201538327e-05, -5.881669266282846e-14, -5.881669266282831e-14]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
