
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_9_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_9", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.959284367287674e+00, -1.298774935927952e+00, -2.363127067401932e-01, -1.800973294857226e-01, -5.226056027109279e-02, -1.019014836736854e-02, -1.875396168608170e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_9_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_9", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.756509080266180e+00, -2.759107634627266e+00, -1.929462055986213e+00, -1.931079893863838e+00, -3.179775445800010e-01, -3.183893009810371e-01, -2.480897398268911e-01, -1.167764670132975e-02, -7.656463028999357e-02, -3.703035474990355e-04, -1.227797010591758e-02, -1.219064446190796e-02, -2.473413997592014e-04, -1.815063090232066e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_9_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_9", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.494702127038297e-04, 0.000000000000000e+00, -5.475894137078672e-04, -2.070533173891413e-03, 0.000000000000000e+00, -2.064953662169988e-03, -3.850747246302849e-02, 0.000000000000000e+00, -4.024424873112979e-02, -8.581104948806210e+00, 0.000000000000000e+00, -3.309680055488921e+01, -6.150162494547105e+01, 0.000000000000000e+00, -8.300871020170818e+04, -6.162889019531281e-01, 0.000000000000000e+00, -2.958995884758614e+01, -1.256993412681115e+00, 0.000000000000000e+00, -1.688962498902758e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_9_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_9", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.072956887972267e-02, 2.070474294378454e-02, 2.904789784503580e-02, 2.903410752565408e-02, 5.190433336508568e-04, 6.227278107557730e-04, 2.243062709194857e-01, 4.230828302786042e-04, 5.763844127127263e-02, 3.382076565423358e-05, 9.152801191215210e-06, 4.303375912078528e-04, 1.526191804997700e-10, -4.660486240030425e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
