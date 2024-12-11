
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b3lyp_mcm1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3lyp_mcm1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.480932947291285e+00, -1.062754141867374e+00, -2.999857652036484e-01, -1.222708034441791e-01, -5.839288924668003e-02, -9.074765759617065e-02, -3.593090332241254e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b3lyp_mcm1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3lyp_mcm1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.860284227534911e+00, -1.861827048228171e+00, -1.288222039558226e+00, -1.289153998998005e+00, -3.841610868870858e-01, -3.845297895600699e-01, -1.583200838599123e-01, -9.724020759340798e-02, -5.521153147448486e-02, -2.582943641521480e-02, -2.665673656948525e-02, -2.685032078341234e-02, -4.933364245023722e-03, -4.346874782099597e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b3lyp_mcm1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3lyp_mcm1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.762910925718257e-04, 5.945130794693804e-06, -1.758243731638552e-04, -6.452319347617600e-04, 4.151313838701667e-05, -6.438118855320732e-04, -3.754324112534190e-02, 5.433974267620157e-02, -3.734029976063204e-02, -2.958746167284937e+00, 5.231780208068396e+00, -8.948053142751885e+02, -5.138770393122861e+01, 2.682904499587453e+01, -3.254066473315559e+07, -7.814626877046618e+02, 9.033659581379508e-02, -7.827095703385505e+02, -9.660975005152412e+07, 0.000000000000000e+00, -2.877900170066257e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
