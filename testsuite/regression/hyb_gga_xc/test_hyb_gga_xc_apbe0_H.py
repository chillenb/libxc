
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_apbe0_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_apbe0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.985730075229526e-01, -4.525220075015733e-01, -2.804110109003481e-01, -1.027933900893473e-01, -5.547765410150484e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_apbe0_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_apbe0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.582069502444133e-01, 1.944572352788634e+00, -5.767058309827288e-01, 8.160774889990077e+01, -3.239591961440090e-01, 4.043302375340972e+01, -1.088930838526562e-01, 2.507828273734719e-01, -7.392633202071123e-03, 7.606532685838403e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_apbe0_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_apbe0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.416365023286555e-03, 3.875923136833987e-02, 1.937961568416993e-02, -1.016350958995775e-02, 2.129528870385742e-02, 1.064764435192871e-02, -1.043305116820028e-01, 7.936241706579927e-02, 3.968120853289963e-02, -3.165954865696505e+00, 1.302054835223529e-01, 6.510274176117674e-02, -3.505879566492174e+00, 1.415406144427693e-03, 7.077030722808280e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
