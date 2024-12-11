
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_apbeint_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_apbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.634463860534681e+01, 8.111633244890548e+00, 6.387888109917754e-01, 1.318878369872935e-01, 2.649454721264879e-02, 1.232370526688642e-03, 4.381281428667161e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_apbeint_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_apbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.593320584694168e+01, 2.598081569870889e+01, 1.229645832991399e+01, 1.231781008223680e+01, 8.259163343552550e-01, 8.259125935928092e-01, 2.137791140996131e-01, 1.869414778250361e-03, 3.351812952520139e-02, 1.882861029887014e-06, 2.066876498472942e-03, 2.037063769960733e-03, 8.400350322142691e-07, 4.245464347148940e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_apbeint_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_apbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.200305164042871e-03, 0.000000000000000e+00, 2.194407378057661e-03, 6.384461124382112e-03, 0.000000000000000e+00, 6.368831154903886e-03, 1.158809687459474e-01, 0.000000000000000e+00, 1.155236360199434e-01, 2.971291692659688e+00, 0.000000000000000e+00, 1.460697212577467e-02, 2.328160833053021e+01, 0.000000000000000e+00, 2.962476035968338e-03, 1.560978603710308e-02, 0.000000000000000e+00, 1.447054309074691e-02, 1.440466436010077e-03, 0.000000000000000e+00, 1.465807780094330e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
