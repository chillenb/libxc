
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_n12_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_n12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-8.205154233966171e-02, -6.891484842912571e-02, -3.067135919527275e-02, -1.593200067077846e-02, -1.608772125916255e-02, -4.940682437089466e-02, -9.827924681297265e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_n12_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_n12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.315153893117718e-01, -1.314894299098851e-01, -7.336176295617897e-02, -7.324057538662827e-02, 9.066103540180231e-02, 9.117581328588494e-02, -3.369552378260697e-02, -1.517648549272852e+00, -7.072179125900014e-03, -9.332147526653928e-01, -6.165104400309802e-02, -6.310521582141340e-02, -8.584756217424165e-04, -2.546300658147900e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_n12_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_n12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([7.104210768118178e-05, 0.000000000000000e+00, 7.106469420431080e-05, -1.478387168996307e-05, 0.000000000000000e+00, -1.450703404286436e-05, -6.101175047684263e-02, 0.000000000000000e+00, -6.115624432516668e-02, 7.615032951770335e+00, 0.000000000000000e+00, -4.325726266861961e+02, -2.662012997189927e+01, 0.000000000000000e+00, -5.121362361650324e+04, -5.303795357663629e+00, 0.000000000000000e+00, -5.618100130033513e+00, -9.125359765578933e+00, 0.000000000000000e+00, -1.359387268214149e+02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
