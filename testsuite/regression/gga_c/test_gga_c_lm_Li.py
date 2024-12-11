
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_lm_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_lm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.123190226142526e-01, -1.090231569441847e-01, -1.433613036587079e-01, -3.463558239574508e-02, -2.223090712409203e-02, -2.677497384203924e+00, -6.346602732249167e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_lm_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_lm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-9.168578571204707e-02, -9.154792117111578e-02, -6.483017119645411e-02, -6.471133878939649e-02, 1.088099502084666e-01, 1.092820413935218e-01, -3.984211888999379e-02, 3.570034742405744e+00, -1.884481048616603e-02, 4.602251668454475e+01, 3.470399784293399e+00, 3.618315755301994e+00, 8.086462143057256e+01, 9.507485838829696e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_lm_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_lm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.314867270774172e-05, 5.249447410255195e-05, -8.275528458408639e-05, -3.935934822375081e-04, 1.544004677518268e-04, -3.919940364815661e-04, -1.426774950470080e-01, -2.647910698504877e-03, -1.429182055717791e-01, 1.200705088223073e+00, 5.671413998413136e+00, -7.091723372939204e+04, -1.742919187134173e+01, 9.754812497950779e+01, -7.009532226533040e+10, -5.800798656255004e+04, -1.818308860581963e+01, -5.972644112428057e+04, -3.521539660705650e+11, -8.900557452091565e-10, -1.378724277754027e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
