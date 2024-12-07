
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_pbe_sol0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe_sol0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.394163890312884e+00, -9.893131439338999e-01, -2.925410837068190e-01, -1.350143488547600e-01, -5.806158387717028e-02, -1.540117309993545e-02, -2.878939130213583e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_pbe_sol0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe_sol0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.820901719627262e+00, -1.822354932564304e+00, -1.268944861316667e+00, -1.269871100788387e+00, -2.965183115991992e-01, -2.965426129542877e-01, -1.786752931913843e-01, -1.256844130336312e-01, -6.562599534549686e-02, 4.664061216607269e-01, -2.056347790911001e-02, -2.041711419679059e-02, -4.156161266258057e-04, -2.954654277202355e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_pbe_sol0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe_sol0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.138291075137597e-05, 8.057094003003931e-05, -7.099104968632221e-05, -3.206390575570424e-04, 2.747845524533765e-04, -3.191333637906462e-04, -4.936763287281101e-02, 9.764292986584010e-03, -4.929286422360783e-02, 7.866101269202956e-01, 4.978936063600431e+00, 2.119832755260273e+00, -2.452266741692966e+01, 3.188247312456189e+01, 1.357225358900635e+01, -3.751920898541450e-01, 7.058310792173810e-04, -3.503758508784056e-01, -1.724534135822350e+00, 6.759926358451161e-06, -2.468496472323308e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
