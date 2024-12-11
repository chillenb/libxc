
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_wb97x_v_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97x_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.419274906766292e+00, -9.807441451826093e-01, -2.356275551854288e-01, -4.863213995084083e-02, -8.885412985325235e-03, 9.210905483632903e-04, 4.238157878048220e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_wb97x_v_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97x_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.834708524294269e+00, -1.836256897702709e+00, -1.217634680458817e+00, -1.218571249253720e+00, -2.026956212147400e-01, -2.029660060376542e-01, -8.082330660017370e-02, 7.281632719497692e-02, -1.099165218540695e-02, 4.492030108640133e-02, 1.093544386938875e-03, 1.177729259755320e-03, -1.992398550723846e-05, 7.842796822672011e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_wb97x_v_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97x_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.457132056458144e-04, 0.000000000000000e+00, -1.452435337952100e-04, -6.372758292328261e-04, 0.000000000000000e+00, -6.354752923531289e-04, -7.708180360886424e-02, 0.000000000000000e+00, -7.691569440886481e-02, 6.075701390392885e-01, 0.000000000000000e+00, 1.077732141528009e+01, -8.059453656522455e+00, 0.000000000000000e+00, 1.267740249440795e+03, 1.281505759434331e-01, 0.000000000000000e+00, 1.362433097408318e-01, 2.236222309411136e-01, 0.000000000000000e+00, 3.361693279026944e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
