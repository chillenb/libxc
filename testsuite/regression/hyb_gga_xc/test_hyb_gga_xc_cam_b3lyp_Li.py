
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_cam_b3lyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.435657892492902e+00, -1.012179471040247e+00, -2.341453365964142e-01, -7.970461878395260e-02, -3.237724229488886e-02, -4.960855320963862e-02, -1.882018384964290e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_cam_b3lyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.810868346637321e+00, -1.812407245578579e+00, -1.233779674442883e+00, -1.234707392803685e+00, -3.263650204861978e-01, -3.266412262782189e-01, -1.105569761958850e-01, -1.036630625636044e-01, -3.172657107580328e-02, -4.137907289658152e-02, -1.665807935272181e-02, -1.681089985870628e-02, -2.664648631612824e-03, -2.388227092774256e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_cam_b3lyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.969794368718018e-04, 4.230480491699886e-06, -1.964278943412853e-04, -6.968371741788078e-04, 2.954022849291356e-05, -6.951673554027792e-04, -2.471328531216479e-02, 3.866747919504811e-02, -2.455254527323596e-02, -1.746651109271521e+00, 3.722869163256963e+00, -4.660631315414244e+02, -2.692553787261530e+01, 1.909121184807025e+01, -1.697604723954086e+07, -4.076703435679776e+02, 6.428238830639903e-02, -4.083207791051115e+02, -5.040007827997233e+07, 0.000000000000000e+00, -1.501363928339826e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
